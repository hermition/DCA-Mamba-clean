"""在子进程导出原实现基线，再验证独立精简实现；不写入原仓库。"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import uuid
from datetime import datetime


def export_reference(args):
    import numpy as np
    import torch
    import yaml
    import albumentations as A

    # 原 plt_conf 在导入时覆盖 CUDA_VISIBLE_DEVICES；先初始化以保持所选 GPU。
    selected_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(args.device)
    torch.cuda.init()
    sys.path.insert(0, str(args.original_repo.resolve()))
    import val
    from utils.plt_conf import compute_calibration
    if selected_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    config_path = args.checkpoint.resolve().with_name("config.yml")
    config = yaml.safe_load(config_path.read_text())
    config["dataset_dir"] = str(args.data_root.resolve().parent)
    config["dataset"] = args.data_root.resolve().name
    assert (config["input_h"], config["input_w"], config["batch_size"]) == (256, 256, 4)
    model = val.load_model(config).to(args.device).eval()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    transform = A.Compose([A.Resize(256, 256), A.Normalize()])
    _, _, dataset = val.get_skin_dataset(config, transform, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False,
                                        num_workers=0, drop_last=False)
    images, targets, logits, probabilities = [], [], [], []

    class CaptureLoader:
        def __len__(self):
            return len(loader)

        def __iter__(self):
            for batch in loader:
                images.append(batch[0].numpy().copy())
                targets.append(batch[1].numpy().copy())
                yield batch

    def capture(module, inputs, output):
        logits.append(output.detach().cpu().numpy().copy())
        probabilities.append(torch.sigmoid(output).detach().cpu().numpy().copy())

    def calibration_only(y_true, y_probs, save_path, thod=0.5, **kwargs):
        y_pred = y_probs > thod
        negative = y_pred == 0
        y_probs[negative] = 1 - y_probs[negative]
        return compute_calibration(y_true, y_pred, y_probs, 15)["expected_calibration_error"]

    handle = model.register_forward_hook(capture)
    val.draw_conf = calibration_only
    torch.backends.cudnn.benchmark = True
    # 绘图函数仅在内存中替换，原 val_model 和指标函数直接运行。
    metrics = val.val_model(model, CaptureLoader(), config, viz=False,
                            dataset_dir=str(args.data_root.resolve().parent),
                            save_heat_map=False, save_ece=True, save_dir=None, thod=0.3)
    handle.remove()
    args.output_dir.mkdir()
    np.savez(args.output_dir / "tensors.npz", images=np.concatenate(images),
             targets=np.concatenate(targets), logits=np.concatenate(logits),
             probabilities=np.concatenate(probabilities))
    source_files = ["archs.py", "train.py", "val.py", "utils/dataset.py", "utils/metrics.py",
                    "utils/plt_conf.py",
                    "Archs/EfficientVit/efficientvit/models/efficientvit/backboneLocal.py",
                    "Archs/EfficientVit/efficientvit/models/efficientvit/seg.py",
                    "Archs/EfficientVit/efficientvit/models/nn/ops.py",
                    "Archs/EfficientVit/efficientvit/models/nn/vmamba.py",
                    "Archs/EfficientVit/efficientvit/models/nn/CBAM.py",
                    "Archs/EfficientVit/efficientvit/models/nn/DefBlock.py"]
    metadata = dict(metrics=metrics, test_ids=dataset.img_ids,
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    parameters=sum(p.numel() for p in model.parameters()),
                    state_shapes={key: list(value.shape) for key, value in state.items()},
                    source_sha256={name: hashlib.sha256(
                        (args.original_repo / name).read_bytes()).hexdigest()
                        for name in source_files})
    (args.output_dir / "reference.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2))
    print("原实现基线导出完成", json.dumps(metrics), flush=True)


def verify(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from inference import BATCH_SIZE, THRESHOLD, environment, evaluate, load_model, sha256
    from dca_model.data import REFERENCE, RingDataset

    args.output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(args.output_dir / "verify.log"),
                                  logging.StreamHandler(sys.stdout)])
    child_args = [sys.executable, str(Path(__file__).resolve()), "--export-reference",
                  "--original-repo", str(args.original_repo.resolve()),
                  "--checkpoint", str(args.checkpoint.resolve()),
                  "--data-root", str(args.data_root.resolve()),
                  "--output-dir", str(args.output_dir.resolve() / "reference"),
                  "--device", args.device]
    logging.info("导出原实现输入、logits 和十项指标")
    with (args.output_dir / "reference.log").open("w") as log:
        subprocess.run(child_args, cwd=args.original_repo.resolve(),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    golden = json.loads((args.output_dir / "reference/reference.json").read_text())
    assert golden["checkpoint_sha256"] == sha256(args.checkpoint) == REFERENCE["checkpoint_sha256"]
    dataset = RingDataset(args.data_root)
    assert dataset.img_ids == golden["test_ids"] and len(dataset) == 43
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, drop_last=False)
    model = load_model(args.checkpoint, args.device)
    assert {key: list(value.shape) for key, value in model.state_dict().items()} == golden["state_shapes"]
    assert golden["parameters"] == sum(p.numel() for p in model.parameters()) == REFERENCE["parameters"]
    assert len(model.state_dict()) == REFERENCE["state_keys"]
    torch.backends.cudnn.benchmark = True
    batch_sizes, maximum_errors, mask_mismatches = [], [], []

    with np.load(args.output_dir / "reference/tensors.npz") as tensors:
        def compare(index, images, target, meta, output):
            start = index * BATCH_SIZE
            end = start + len(images)
            assert meta["img_id"] == golden["test_ids"][start:end]
            np.testing.assert_array_equal(images.cpu().numpy(), tensors["images"][start:end])
            np.testing.assert_array_equal(target.cpu().numpy(), tensors["targets"][start:end])
            expected = torch.from_numpy(tensors["logits"][start:end]).to(args.device)
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
            probability = torch.sigmoid(output).cpu().numpy()
            difference = int(np.count_nonzero(
                (probability > THRESHOLD) != (tensors["probabilities"][start:end] > THRESHOLD)))
            assert difference == 0, f"batch {index} 掩码像素不一致：{difference}"
            maximum_errors.append(float(torch.max(torch.abs(output - expected))))
            mask_mismatches.append(difference)
            batch_sizes.append(len(images))

        metrics = evaluate(model, loader, args.device, on_batch=compare)
    assert batch_sizes == [4] * 10 + [3]
    differences = {key: abs(metrics[key] - golden["metrics"][key]) for key in metrics}
    for key in metrics:
        assert differences[key] <= 1e-8, (key, metrics[key], golden["metrics"][key])
        assert abs(metrics[key] - REFERENCE["metrics"][key]) <= 1e-8
    original = args.original_repo.resolve()
    imported_from_original = [name for name, module in sys.modules.items()
                              if getattr(module, "__file__", None)
                              and Path(module.__file__).resolve().is_relative_to(original)]
    assert not imported_from_original, imported_from_original
    report = dict(status="passed", checkpoint_sha256=golden["checkpoint_sha256"],
                  parameters=golden["parameters"], state_keys=len(model.state_dict()),
                  test_count=len(dataset), batch_sizes=batch_sizes, threshold=THRESHOLD,
                  max_logit_abs_error=max(maximum_errors), mask_pixel_mismatches=sum(mask_mismatches),
                  original_metrics=golden["metrics"], clean_metrics=metrics,
                  metric_abs_differences=differences,
                  imported_from_original=imported_from_original, environment=environment(),
                  original_source_sha256=golden["source_sha256"],
                  clean_source_sha256={str(path.relative_to(Path(__file__).parent)): sha256(path)
                                       for path in Path(__file__).parent.rglob("*.py")
                                       if "artifacts" not in path.parts and "runs" not in path.parts})
    (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2))
    logging.info("验收通过；最大 logits 误差=%g，掩码差异像素=%d",
                 report["max_logit_abs_error"], report["mask_pixel_mismatches"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--export-reference", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir = (args.output_dir or Path("artifacts") / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])).resolve()
    if args.export_reference:
        export_reference(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
