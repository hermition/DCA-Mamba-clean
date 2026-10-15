# DCA-Mamba：模型与文件夹推理

精简实现仅保留目标模型、推理和一致性验证代码。固定结构为 EfficientVitMamba、
MHSS2DCasCBAM3、四头 CBAM 排序、级联 add 融合及 MLP 混合，深度为
`[1,2,3,3,6]`；输入 256×256，单类别，无深监督。

## 获取代码与配置环境

```bash
git clone https://github.com/hermition/DCA-Mamba-clean.git
cd DCA-Mamba-clean
conda create -n dca-mamba-infer python=3.10.13
conda activate dca-mamba-infer
```

验证环境为 Linux x86_64、RTX 2080 Ti、Python 3.10.13、PyTorch 2.0.1+cu117、
CUDA 11.7、Mamba 1.1.0。完整版本见 `requirements.txt`。
从源码安装 CUDA 扩展前，准备 CUDA Toolkit 11.7（包含 `nvcc`）及 C++ 编译器。

```bash
# 改为实际安装的 CUDA Toolkit 11.7 路径。
export CUDA_HOME=/usr/local/cuda-11.7
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version
g++ --version

# 扩展的安装脚本需要事先能导入 torch、packaging 和 wheel。
python -m pip install pip==23.3.1 setuptools==80.9.0 wheel==0.41.2 \
  packaging==24.2 ninja==1.11.1.1 numpy==1.26.3
python -m pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install --no-build-isolation -r requirements.txt
```

旧版 Mamba 和 causal-conv1d 的源码构建需要 `nvcc >= 11.6`；
`nvidia-smi` 显示的 CUDA 版本不能代替 `nvcc --version`。
安装顺序依据 [PyTorch 历史版本说明](https://pytorch.org/get-started/previous-versions/#v201)、
[Mamba 1.1.0 安装源码](https://github.com/state-spaces/mamba/blob/v1.1.0/setup.py)、
[causal-conv1d 1.1.1 安装源码](https://github.com/Dao-AILab/causal-conv1d/blob/v1.1.1/setup.py)。
依赖版本和运行已在已有环境验证；从零安装流程尚未在新环境完整执行。

如果本机已有验证通过的环境，也可克隆为独立环境：

```bash
conda create -n dca-mamba-infer --clone /path/to/existing/env -y
conda activate dca-mamba-infer
```

### 环境自检

在代码根目录执行，实际检查 CUDA 扩展与前向计算：

```bash
python - <<'PY'
import torch
import torchvision
import mamba_ssm
import causal_conv1d_cuda
import selective_scan_cuda
from dca_model import build_model

print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("torchvision:", torchvision.__version__, "Mamba:", mamba_ssm.__version__)
assert torch.cuda.is_available(), "CUDA 不可用"
model = build_model().cuda().eval()
with torch.no_grad():
    output = model(torch.zeros(1, 3, 256, 256, device="cuda"))
assert output.shape == (1, 1, 256, 256)
assert torch.isfinite(output).all()
print("CUDA 前向自检通过")
PY
```

该自检使用随机初始化参数，只检查环境。正式推理会严格加载下面的权重。

## 准备权重

权重统一使用匿名本地路径 **`./checkpoints/model.pth`**。将实验权重放到该位置，
或通过 `--checkpoint /path/to/model.pth` 指定。仓库不包含权重，也没有配置权重下载链接。

```bash
mkdir -p checkpoints
# 将模型权重保存为 checkpoints/model.pth。
```

权重 SHA256：

```text
81ead141986aa64a8d167c000d58ad843b50f88917bb9cdebb30960dbb68f77c
```

可执行 `sha256sum checkpoints/model.pth` 核对。模型直接以 `strict=True` 加载，
无需转换权重。

## 预测图片文件夹

图片文件夹无需标注，不要求固定图像数量或数据集划分：

```text
images/
├── sample.jpg
├── another.png
└── example.JPEG
```

在代码根目录运行：

```bash
python inference.py --input-dir ./images
```

完整参数示例：

```bash
python inference.py \
  --input-dir /path/to/images \
  --checkpoint ./checkpoints/model.pth \
  --output-dir ./output \
  --device cuda:0
```

只读取指定文件夹的直接子文件，支持 JPG、JPEG、PNG、BMP、TIF、TIFF、WEBP，
扩展名不区分大小写。文件夹不存在、没有支持的图片、图片损坏或缺少权重时会报错。
如设置了 `CUDA_VISIBLE_DEVICES`，`--device cuda:N` 中的 `N` 是可见设备的逻辑编号。

结果默认保存到当前工作目录的 **`./output/<时间戳>_<随机标识>/`**。
每次运行创建新目录，已有结果保留：

```text
output/
└── <时间戳>_<随机标识>/
    ├── sample.jpg.png
    ├── another.png.png
    ├── example.JPEG.png
    ├── inference.log
    └── run.json
```

掩码为 **256×256、0/255 PNG**，阈值固定为 **0.3**；不会缩放回原图大小。
输出名采用“完整输入文件名 + .png”，同名不同扩展名的图片不会互相覆盖。
`run.json` 记录权重 SHA256、图像数、推理配置与环境版本；`inference.log` 保存运行日志。
无标注推理不计算分割指标。

### 使用 screen 后台运行

先激活环境，在代码根目录执行：

```bash
inference_python="$(command -v python)"
mkdir -p output
inference_log="$(mktemp "$PWD/output/screen-XXXXXXXX.log")"
inference_session="dca_infer_$(date +%Y%m%d_%H%M%S)_$$"
screen -L -Logfile "$inference_log" -dmS "$inference_session" \
  "$inference_python" -u "$PWD/inference.py" \
  --input-dir ./images \
  --checkpoint ./checkpoints/model.pth \
  --device cuda:0
echo "screen: $inference_session"
tail -F "$inference_log"
```

执行 `screen -r "$inference_session"` 进入会话，按 `Ctrl+A`、`D` 脱离；
结束后日志和预测结果保留在 `./output` 中。

## 模型接口与一致性验证

```python
import torch
from dca_model import build_model

model = build_model()
model.load_state_dict(torch.load("./checkpoints/model.pth", map_location="cpu"), strict=True)
model = model.cuda().eval()
with torch.no_grad():
    logits = model(images)  # 已按原实验流程预处理的 [B,3,256,256]
    masks = torch.sigmoid(logits) > 0.3
```

预处理保留原实验的 OpenCV BGR、Resize(256,256)、默认 Normalize，
以及 Normalize 后再次除以 255 的实际流程。batch=4，FP32，无 TTA，
模型上采样沿用原 bicubic。文件夹推理不使用固定测试集清单。

保留 `RingDataset` 和 `verify_parity.py`，用于有标注的原 DCA ring 43 张测试图验收。
该可选验证需要另行提供原实现及其完整依赖，并将对应的 `config.yml` 放在验证用权重旁：

```bash
python verify_parity.py \
  --original-repo /path/to/original/repository \
  --checkpoint ./checkpoints/model.pth \
  --data-root /path/to/dca_dataset_ring \
  --device cuda:0
```

正常文件夹推理不导入原仓库，也不需要 `config.yml`、标注或原数据集。
原测试集验收按原顺序和 batch 聚合口径，包括原 HD95 与 15 分箱 ECE。
418 个状态项严格匹配，参数量 22,313,993；已验证 43 张图的 logits 最大绝对误差为 0，
掩码差异像素为 0，十项指标差值全部为 0。报告见 `validation/verification.json`。

| 指标 | 原实现与精简实现 |
|---|---:|
| IoU | 0.7022113659525491 |
| Dice | 0.8200896055839917 |
| Acc | 0.9692407652389171 |
| Precision | 0.8441372989134135 |
| Recall | 0.8170538113944286 |
| SPE | 0.9867809917069728 |
| HD95 | 4.150606272813468 |
| ECE | 0.0236306253407451 |
| F1 | 0.8200896006567144 |
| BS | 0.026822711232789728 |
