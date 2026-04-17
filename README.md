# TBD: To Be Detected

EECS 504 Computer Vision project for still-image distracted driver detection on the State Farm dataset.

## Demo

### Standard Prediction Demo

![Standard prediction demo](./demo_before_multiclass_raw.gif)

### Grad-CAM + MediaPipe Verification Demo

![Grad-CAM + MediaPipe verification demo](./demo_gradcam_multiclass_heuristic.gif)

## Dataset

- Source: [State Farm Distracted Driver Detection (Kaggle)](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
- Expected folders:
  - `imgs/train/c0` ... `imgs/train/c9`
  - `imgs/test`

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Train baseline model:

```bash
python train_resnet50_baseline.py --epochs 8 --batch-size 32
```

2. Plot training metrics:

```bash
python plot_training_metrics.py
```

3. Generate Grad-CAM images:

```bash
python gradcam_resnet50_baseline.py --image-dir imgs/test --max-images 5
```

4. Create prediction demo video:

```bash
python demo_video_resnet50_baseline.py --image-dir imgs/test --max-images 20 --fps 2
```

## Outputs

- Checkpoint: `outputs/resnet50_baseline/best_resnet50.pt`
- Metrics: `outputs/resnet50_baseline/metrics.json`
- Curves: `outputs/resnet50_baseline/training_curves.png`
- Grad-CAM images: `outputs/resnet50_baseline/gradcam/`
- Demo video: `outputs/resnet50_baseline/demo_prediction.mp4`

## Team

- Team name: TBD: To Be Detected