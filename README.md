
  
# FreeSplatter 复现

# ⚙️ Installation

注意，`requirements.txt` 中少了 onnxruntime，已补上
We recommend using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA>=12.1`.
```bash
conda create --name freesplatter python=3.10
conda activate freesplatter
pip install -U pip

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.27.post2

# Install other requirements
pip install -r requirements.txt
```

网络问题，`requirements.txt` 中的一些包要手动 `clone` 然后安装。

```bash
cd submodule
rm -r diff-gaussian-rasterization
rm -r diff-surfel-rasterization
rm -r nvdiffrast

git clone --recurse-submodules git@github.com:ashawkey/diff-gaussian-rasterization.git

git clone --recurse-submodules git@github.com:hbb1/diff-surfel-rasterization.git

git clone --recurse-submodules git@github.com:NVlabs/nvdiffrast.git
```

# 🤖 Pretrained Models

We provide the following pretrained models:

| Model | Description | #Params | Download |
| --- | --- | --- | --- |
| FreeSplatter-O | Object-level reconstruction model | 306M | [Download](https://huggingface.co/TencentARC/FreeSplatter/blob/main/freesplatter-object.safetensors) |
| FreeSplatter-O-2dgs | Object-level reconstruction model using [2DGS](https://surfsplatting.github.io/) (finetuned from FreeSplatter-O) | 306M | [Download](https://huggingface.co/TencentARC/FreeSplatter/blob/main/freesplatter-object-2dgs.safetensors) |
| FreeSplatter-S | Scene-level reconstruction model | 306M | [Download](https://huggingface.co/TencentARC/FreeSplatter/blob/main/freesplatter-scene.safetensors) |

# 💫 Inference

We recommand to start a gradio demo in your local machine, simply run:
```bash
python app.py
```