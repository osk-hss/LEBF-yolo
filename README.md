# LEBF-YOLO

![Ultralytics 8.3.9](https://img.shields.io/badge/Ultralytics-8.3.9-blue.svg)
![License AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)

LEBF-YOLO is a YOLO11-based object detection model for small objects in UAV imagery. Built on Ultralytics 8.3.9, it combines four components:

- **C3k2_LFEM** for feature extraction in the backbone, using a fixed 5 × 5 Gaussian kernel with σ = 0.5. Gaussian convolution weights remain frozen throughout training; the following normalization and feature-enhancement layers remain trainable.
- **BiFPN fusion** for weighted multi-scale feature aggregation.
- **Detect_LSDDH** as the shared detail-enhanced detection head.
- **CIWIoU** for bounding-box regression, using CIoU with non-monotonic Wise-IoU focusing.

The model predicts at **P2, P3, and P4**, with strides of 4, 8, and 16.

## 📦 Environment and Installation

### Experimental environment

The experiments reported in Section 4.1 of the paper used the following environment:

| Component | Configuration |
| --- | --- |
| Operating system | Linux |
| Python | 3.9.21 |
| PyTorch | 2.6.0 |
| Framework | Ultralytics 8.3.9 with the LEBF-YOLO modifications in this repository |

### Installation on Linux

With Conda installed, create and activate an environment:

```bash
conda create -n lebfyolo python=3.9.21 -y
conda activate lebfyolo
```

Install PyTorch 2.6.0 and its matching torchvision 0.21.0 package for CUDA 12.4, using the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v260):

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

Install the remaining dependencies directly; this installation procedure does not require a `requirements.txt` file:

```bash
python -m pip install "numpy>=1.23.0" "matplotlib>=3.3.0" \
    "opencv-python>=4.6.0" "pillow>=7.1.2" "pyyaml>=5.3.1" \
    "requests>=2.23.0" "scipy>=1.4.1" "tqdm>=4.64.0" \
    psutil py-cpuinfo "pandas>=1.1.4" "seaborn>=0.11.0" \
    timm==1.0.19 einops==0.8.1 "ultralytics-thop>=2.0.0"
```

The paper does not specify the versions of these additional dependencies. The timm and einops pins above are installation choices for this release, rather than versions reported in the paper; this is not a complete lockfile of the original experiment environment.

Download and extract this repository, then open a terminal in its root directory (the directory containing this README and the `ultralytics` folder). Make the local source available to Python in the current shell:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Run the training, validation, and prediction Python examples below from this directory. This procedure uses the modified source in this repository directly.

### Verify the environment

```bash
python -m pip check
python - <<'PY'
import sys
import torch
import torchvision
import ultralytics

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unavailable")
print("Ultralytics:", ultralytics.__version__)
print("Source:", ultralytics.__file__)
PY
```

For the paper's GPU environment, check Python 3.9.21, PyTorch 2.6.0 (possibly with a `+cu124` suffix), CUDA runtime 12.4, and cuDNN 9.1.x. `CUDA available` should be `True`. The Ultralytics version should be `8.3.9`, and its source path should point to this repository. A working NVIDIA driver compatible with the CUDA 12.4 runtime is required for GPU training.


## 🚀 Usage

### Train

Edit the `data` argument in [train.py](train.py) to point to your dataset YAML, then run:

```bash
python train.py
```

The supplied script uses a server-specific dataset path. Update it before training on another machine.

Alternatively, use the Python API:

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/11/yolo11-C3K2-LFEM-BPFN-CIWIOU-LSDDH.yaml")
    model.train(
        data="path/to/your/data.yaml",
        cache=False,
        imgsz=640,
        epochs=200,
        batch=8,
        pretrained=False,
        device=0,
        optimizer="SGD",
        amp=False,
        project="runs/train",
        name="lebfyolo",
    )
```

These settings match the supplied training script. The model configuration defaults to the `n` scale when no scale is specified. The dataset YAML determines the class count during training.

CIWIoU is enabled in `BboxLoss` in [ultralytics/utils/loss.py](ultralytics/utils/loss.py), with `ltype='CIoU'`, `monotonous=False`, `inner_iou=False`, and `focaler_iou=False`. This is a global loss setting in this project; account for it when running baseline comparisons.

### Validate

Load the checkpoint saved by your training run and evaluate it on the validation split:

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("path/to/best.pt")
    model.val(data="path/to/your/data.yaml", imgsz=640, device=0)
```

### Predict

Run inference on an image or a directory of images:

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("path/to/best.pt")
    model.predict(source="path/to/images", imgsz=640, device=0, save=True)
```

Replace the example dataset, checkpoint, and image paths with your own paths.

## 📁 Key Directory Structure

### Configuration Files

- **Model architecture:** [ultralytics/cfg/models/11/LEBF-YOLO.yaml](ultralytics/cfg/models/11/yolo11-C3K2-LFEM-BPFN-CIWIOU-LSDDH.yaml).
- **Dataset configurations:** [ultralytics/cfg/datasets](ultralytics/cfg/datasets), including the VisDrone configuration.
- **Default training settings:** [ultralytics/cfg/default.yaml](ultralytics/cfg/default.yaml).

### Module Implementation

| Component | Implementation |
| --- | --- |
| C3k2_LFEM and BiFPN fusion | [ultralytics/nn/extra_modules/block.py](ultralytics/nn/extra_modules/block.py) |
| Detail-enhanced convolution | [ultralytics/nn/extra_modules/deconv.py](ultralytics/nn/extra_modules/deconv.py) |
| Detect_LSDDH | [ultralytics/nn/extra_modules/head.py](ultralytics/nn/extra_modules/head.py) |
| WiseIoU implementation | [ultralytics/utils/metrics.py](ultralytics/utils/metrics.py) |
| CIWIoU training integration | [ultralytics/utils/loss.py](ultralytics/utils/loss.py) |

## 📊 Dataset

The supplied training script targets **VisDrone2019-DET**, which has 10 detection classes:

`pedestrian`, `people`, `bicycle`, `car`, `van`, `truck`, `tricycle`, `awning-tricycle`, `bus`, and `motor`.

Prepare the images and annotations in YOLO detection format. Each label row contains:

```text
class_id x_center y_center width height
```

Class IDs are zero-based, and coordinates are normalized to the image dimensions. Use corresponding `images` and `labels` directories, for example:

```text
VisDrone/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── labels/
└── VisDrone2019-DET-val/
    ├── images/
    └── labels/
```

Use [ultralytics/cfg/datasets/VisDrone.yaml](ultralytics/cfg/datasets/VisDrone.yaml) as a reference for the dataset root, split paths, and class names. Point the `data` argument in training and validation to your prepared dataset YAML.

## 📄 License and Acknowledgments

The repository includes the [AGPL-3.0 license](LICENSE).

This project is built on Ultralytics YOLO11. We thank the Ultralytics developers and the contributors to the underlying modules.
