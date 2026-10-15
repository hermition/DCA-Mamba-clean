# DCA-Mamba

## 1. 推理示例

将权重放到 `./checkpoints/model.pth`，运行：

```bash
python inference.py --input-dir ./images --checkpoint ./checkpoints/model.pth
```

预测结果保存在 `./output/<时间戳>_<随机标识>/`，每次运行保留旧结果。

## 2. 配置环境

使用 Linux 和 CUDA Toolkit 11.7，在代码根目录执行：

```bash
conda create -n dca-mamba-infer python=3.10.13 -y
conda activate dca-mamba-infer
python -m pip install pip==23.3.1 setuptools==80.9.0 wheel==0.41.2 packaging==24.2 ninja==1.11.1.1 numpy==1.26.3
python -m pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install --no-build-isolation -r requirements.txt
```
