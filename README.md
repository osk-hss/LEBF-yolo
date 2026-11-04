# LEBF-YOLO

![Python 3.9.21](https://img.shields.io/badge/Python-3.9.21-green.svg)
![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)

LEBF-YOLO is a YOLO11-based model for small-object detection in UAV imagery, incorporating C3k2-LFEM, BiFPN fusion, LSDDH, and CIWIoU.

## 📦 Installation

The paper uses Linux, Python 3.9.21, PyTorch 2.6.0, CUDA 12.4, cuDNN 9.1, and an NVIDIA RTX 3090 GPU (24 GB).

Create the environment and install dependencies:

```bash
conda create -n lebfyolo python=3.9.21 -y
conda activate lebfyolo
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install "numpy>=1.23.0" "matplotlib>=3.3.0" \
    "opencv-python>=4.6.0" "pillow>=7.1.2" "pyyaml>=5.3.1" \
    "requests>=2.23.0" "scipy>=1.4.1" "tqdm>=4.64.0" \
    psutil py-cpuinfo "pandas>=1.1.4" "seaborn>=0.11.0" \
    timm==1.0.19 einops==0.8.1 "ultralytics-thop>=2.0.0"
```

Download this repository and run the examples from its root directory to use the local modified Ultralytics 8.3.9 code. No `requirements.txt` is needed. The additional dependency versions above are installation settings, not a complete record of the paper's environment.

## 🚀 Usage

Replace the dataset, checkpoint, and image paths with your own paths.

### Train

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/11/yolo11-C3K2-LFEM-BPFN-CIWIOU-LSDDH.yaml")
    model.train(
        data="path/to/data.yaml", epochs=200, imgsz=640, batch=8,
        optimizer="SGD", pretrained=False, amp=False, device=0,
    )
```

CIWIoU is enabled globally in `ultralytics/utils/loss.py`; disable it when training a baseline without this improvement.

### Validate

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("path/to/best.pt")
    model.val(data="path/to/data.yaml", imgsz=640, device=0)
```

### Predict

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("path/to/best.pt")
    model.predict(source="path/to/images", imgsz=640, device=0, save=True)
```

## 📁 Key Directory Structure

- **Model configuration:** [ultralytics/cfg/models/11/yolo11-C3K2-LFEM-BPFN-CIWIOU-LSDDH.yaml](ultralytics/cfg/models/11/yolo11-C3K2-LFEM-BPFN-CIWIOU-LSDDH.yaml)
- **Dataset configurations:** [ultralytics/cfg/datasets](ultralytics/cfg/datasets)
- **Training settings:** [ultralytics/cfg/default.yaml](ultralytics/cfg/default.yaml)
- **C3k2-LFEM and BiFPN:** [ultralytics/nn/extra_modules/block.py](ultralytics/nn/extra_modules/block.py)
- **LSDDH:** [ultralytics/nn/extra_modules/head.py](ultralytics/nn/extra_modules/head.py) and [deconv.py](ultralytics/nn/extra_modules/deconv.py)
- **CIWIoU:** [ultralytics/utils/metrics.py](ultralytics/utils/metrics.py) and [loss.py](ultralytics/utils/loss.py)

## 📊 Dataset

The paper evaluates LEBF-YOLO on **VisDrone2019-DET**, **SkyFusion**, and **RS-STOD**. Prepare images and annotations in YOLO detection format, and set the dataset paths and class names in your dataset YAML.

For VisDrone2019-DET, refer to [ultralytics/cfg/datasets/VisDrone.yaml](ultralytics/cfg/datasets/VisDrone.yaml).

## 📄 License

This project is based on Ultralytics YOLO11 and distributed under the [AGPL-3.0 license](LICENSE).
