import hashlib
import json
from glob import glob
from pathlib import Path

import albumentations as A
import cv2
from torch.utils.data import Dataset


REFERENCE = json.loads(Path(__file__).with_name("reference.json").read_text())


class ImageFolderDataset(Dataset):
    """读取文件夹中的图片；不需要标注或固定测试集。"""

    def __init__(self, input_dir):
        self.root = Path(input_dir).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"图片文件夹不存在：{input_dir}")
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        self.paths = sorted(path for path in self.root.iterdir()
                            if path.is_file() and path.suffix.lower() in extensions)
        if not self.paths:
            raise ValueError(f"文件夹中没有支持的图片：{input_dir}")
        self.transform = A.Compose([A.Resize(256, 256), A.Normalize()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"无法读取图片：{path.name}")
        image = self.transform(image=image)["image"]
        # 原 Dataset 在 Normalize 后再次 /255；为兼容旧权重必须保留。
        image = (image.astype("float32") / 255).transpose(2, 0, 1)
        return image, {"filename": path.name}


class RingDataset(Dataset):
    """按原实验顺序读取 DCA ring test，保持原来的 BGR 和双重缩放。"""

    def __init__(self, data_root):
        self.root = Path(data_root).resolve() / "test"
        paths = glob(str(self.root / "imgs" / "*.jpg"))
        by_hash = {hashlib.sha256(Path(p).stem.encode()).hexdigest(): Path(p).stem
                   for p in paths}
        expected = REFERENCE["test_order_sha256"]
        if len(paths) != len(expected) or set(by_hash) != set(expected):
            raise ValueError("test 图像与原实验的 43 张图像不匹配")
        self.img_ids = [by_hash[key] for key in expected]
        self.transform = A.Compose([A.Resize(256, 256), A.Normalize()])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        image_path = self.root / "imgs" / (img_id + ".jpg")
        mask_path = self.root / "masks" / "0" / (img_id + ".png")
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"无法读取图像或标注：{image_path}，{mask_path}")
        augmented = self.transform(image=image, mask=mask[..., None])
        # 原 Dataset 在 Normalize 后再次 /255；为兼容旧权重必须保留。
        image = (augmented["image"].astype("float32") / 255).transpose(2, 0, 1)
        mask = (augmented["mask"].astype("float32") / 255).transpose(2, 0, 1)
        return image, mask, {"img_id": img_id}
