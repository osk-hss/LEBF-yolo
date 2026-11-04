# LEBF-YOLO

![Ultralytics 8.3.9](https://img.shields.io/badge/Ultralytics-8.3.9-blue.svg)
![License AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)

LEBF-YOLO is a YOLO11-based object detection model for small objects in UAV imagery. Built on Ultralytics 8.3.9, it combines four components:

- **C3k2_LFEM** for feature extraction in the backbone, using a fixed 5 × 5 Gaussian kernel with σ = 0.5. Gaussian convolution weights remain frozen throughout training; the following normalization and feature-enhancement layers remain trainable.
- **BiFPN fusion** for weighted multi-scale feature aggregation.
- **Detect_LSDDH** as the shared detail-enhanced detection head.
- **CIWIoU** for bounding-box regression, using CIoU with non-monotonic Wise-IoU focusing.

The model predicts at **P2, P3, and P4**, with strides of 4, 8, and 16.

## 📦 Installation

From the project root, install the local package and its dependencies:

```bash
python -m pip install -e .
```

Dependencies are defined in [pyproject.toml](pyproject.toml), including PyTorch, torchvision, timm, and einops. Use a PyTorch installation compatible with your CUDA environment for GPU training.

Check that Python imports this project:

```bash
python -c "import ultralytics; print(ultralytics.__version__); print(ultralytics.__file__)"
```

The version should be `8.3.9`, and the package path should point to this repository.

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
