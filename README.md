# LEBF-YOLO

![Python 3.9.21](https://img.shields.io/badge/Python-3.9.21-green.svg)
![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)

LEBF-YOLO is a YOLO11-based model for small-object detection in UAV imagery, incorporating C3k2-LFEM, BiFPN fusion, LSDDH, and CIWIoU.

## 📦 Installation

Install the dependencies in a **Python 3.9.21** environment:

```bash
pip install torch==2.6.0 torchvision numpy matplotlib opencv-python pillow pyyaml requests scipy tqdm psutil py-cpuinfo pandas seaborn timm einops ultralytics-thop
```

This installs **PyTorch 2.6.0** and the dependencies required by this project.

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
