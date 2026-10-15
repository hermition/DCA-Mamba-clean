import argparse
import hashlib
import importlib.metadata
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dca_model import build_model
from dca_model.data import ImageFolderDataset
from dca_model.metrics import AverageMeter, acc_score, compute_calibration, iou_score


THRESHOLD = 0.3
BATCH_SIZE = 4


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    names = ["torch", "timm", "mamba-ssm", "causal-conv1d", "albumentations",
             "opencv-python", "numpy", "einops", "MedPy", "scipy"]
    return {"python": sys.version.split()[0], "cuda": torch.version.cuda,
            "packages": {name: importlib.metadata.version(name) for name in names}}


def load_model(checkpoint, device):
    model = build_model()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    logging.info("strict=True 加载成功；参数=%d，状态项=%d",
                 sum(p.numel() for p in model.parameters()), len(state))
    return model.to(device).eval()


def calibration(y_true, y_probs):
    # 沿用 draw_conf 的 15 分箱及背景置信度转换，避免引入绘图依赖。
    y_pred = y_probs > THRESHOLD
    negative = y_pred == 0
    confidence = y_probs.copy()
    confidence[negative] = 1 - confidence[negative]
    return compute_calibration(y_true, y_pred, confidence, 15)["expected_calibration_error"]


def evaluate(model, loader, device, mask_dir=None, on_batch=None):
    names = ["IoU", "Dice", "Acc", "Precision", "Recall", "SPE", "HD95", "F1", "BS"]
    meters = {name: AverageMeter() for name in names}
    probs, targets = [], []
    with torch.no_grad():
        for index, (images, target, meta) in enumerate(loader):
            images, target = images.to(device), target.to(device)
            logits = model(images)
            if logits.shape != target.shape or not torch.isfinite(logits).all():
                raise ValueError(f"无效输出：{tuple(logits.shape)}")
            if on_batch is not None:
                on_batch(index, images, target, meta, logits)
            iou, dice = iou_score(logits, target, thod=THRESHOLD)
            acc, f1, precision, recall, spe, distance, _, bs = acc_score(
                logits, target, thod=THRESHOLD)
            values = [iou, dice, acc, precision, recall, spe, distance, f1, bs]
            for name, value in zip(names, values):
                meters[name].update(value, images.size(0))
            probability = torch.sigmoid(logits).cpu().numpy()
            probs.append(probability.flatten())
            targets.append(target.cpu().numpy().flatten())
            if mask_dir is not None:
                for img_id, mask in zip(meta["img_id"], probability[:, 0] > THRESHOLD):
                    path = mask_dir / (img_id + ".png")
                    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
                        raise OSError(f"预测掩码保存失败：{path}")
            logging.info("batch %d/%d，图像数=%d", index + 1, len(loader), images.size(0))
    result = {name: float(meter.avg) for name, meter in meters.items()}
    result["ECE"] = float(calibration(np.concatenate(targets), np.concatenate(probs)))
    return result


def main():
    parser = argparse.ArgumentParser(description="预测图片文件夹，阈值 0.3，默认输出到 ./output")
    parser.add_argument("--input-dir", required=True, type=Path, help="包含图片的文件夹，无需标注")
    parser.add_argument("--checkpoint", default=Path("checkpoints/model.pth"), type=Path)
    parser.add_argument("--output-dir", default=Path("output"), type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("原 selective_scan 算子需要 CUDA，请使用 --device cuda:N")
    dataset = ImageFolderDataset(args.input_dir)
    if not args.checkpoint.is_file():
        parser.error(f"权重不存在：{args.checkpoint}；请将权重放到 ./checkpoints/model.pth")
    output = args.output_dir / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(output / "inference.log"),
                                  logging.StreamHandler(sys.stdout)])
    try:
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, drop_last=False)
        model = load_model(args.checkpoint, args.device)
        torch.backends.cudnn.benchmark = True
        metadata = dict(checkpoint=args.checkpoint.name,
                        checkpoint_sha256=sha256(args.checkpoint),
                        threshold=THRESHOLD,
                        batch_size=BATCH_SIZE, input_size=[256, 256], precision="float32",
                        device=args.device, image_count=len(dataset), environment=environment())
        (output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        with torch.no_grad():
            for index, (images, meta) in enumerate(loader):
                logits = model(images.to(args.device))
                if logits.shape != (len(images), 1, 256, 256) or not torch.isfinite(logits).all():
                    raise ValueError(f"无效输出：{tuple(logits.shape)}")
                masks = (torch.sigmoid(logits) > THRESHOLD).cpu().numpy()[:, 0]
                for filename, mask in zip(meta["filename"], masks):
                    path = output / (filename + ".png")
                    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
                        raise OSError(f"预测掩码保存失败：{path}")
                logging.info("batch %d/%d，图像数=%d", index + 1, len(loader), len(images))
        logging.info("输出目录：%s", output.resolve())
    except Exception:
        logging.exception("推理失败")
        raise


if __name__ == "__main__":
    main()
