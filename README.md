# TBD: To Be Detected

EECS 504 Computer Vision course project on still-image distracted driver detection using the State Farm Distracted Driver Detection dataset.


## Overview

This project studies whether a vision model can detect distracted driving behavior from a single image and which visual regions matter most for that decision. We compare full-image classification against driver-centered crops and tighter crops around the face, hands, and phone. We also compare binary distracted-versus-safe classification with fine-grained multiclass classification and use Grad-CAM to verify whether the model attends to meaningful visual evidence.

![output_DEb8oT](https://github.com/user-attachments/assets/a963001a-8e6e-4d91-aabb-f2edf1384dbd)


## Motivation

Distracted driving is a major safety issue that contributes to preventable traffic accidents. Our goal is to explore how computer vision can support safer driving through image-based driver monitoring while keeping the system interpretable enough to understand what visual cues the model is using.

## Dataset

- Dataset: State Farm Distracted Driver Detection
- Source: Kaggle competition data
- Main metadata file: [`state-farm-distracted-driver-detection/driver_imgs_list.csv`](./state-farm-distracted-driver-detection/driver_imgs_list.csv)
- Submission format reference: [`state-farm-distracted-driver-detection/sample_submission.csv`](./state-farm-distracted-driver-detection/sample_submission.csv)
- Image folders:
  - Training: [`state-farm-distracted-driver-detection/imgs/train`](./state-farm-distracted-driver-detection/imgs/train)
  - Test: [`state-farm-distracted-driver-detection/imgs/test`](./state-farm-distracted-driver-detection/imgs/test)

The training set contains 10 classes (`c0` through `c9`) and subject identifiers for each image. Because the same driver appears in multiple images, evaluation should use subject-aware splitting rather than a random image split.

## Project Questions

1. How well can a model detect distracted driving from a single still image?
2. Does focusing on driver-centered regions improve performance over full-image classification?
3. Do face, hand, and phone crops provide better visual evidence than the entire frame?
4. Is binary classification (`safe` vs `distracted`) more robust than fine-grained multiclass classification?
5. Does the model attend to semantically meaningful regions according to Grad-CAM?

## Planned Tasks

### 1. Baseline Classification

- Train a full-image multiclass classifier on the original 10 classes
- Train a full-image binary classifier with `c0` as `safe` and `c1-c9` as `distracted`

### 2. Region-Based Experiments

- Compare full-image inputs against driver-region crops
- Compare driver-region crops against tighter face/hand/phone-focused crops when available

### 3. Explainability

- Use Grad-CAM to visualize which regions contribute most to each prediction
- Check whether the model focuses on the driver, hands, and phone instead of irrelevant background cues

## ResNet-50 Baseline (Implemented)

- Script: `train_resnet50_baseline.py`
- Input format: folder-based classes under `imgs/train` (`c0` to `c9`)
- Output:
  - best checkpoint: `outputs/resnet50_baseline/best_resnet50.pt`
  - training history: `outputs/resnet50_baseline/metrics.json`

### Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run a short smoke test (fast sanity check):

```bash
python train_resnet50_baseline.py --epochs 1 --batch-size 8 --num-workers 0 --max-train-batches 5 --max-val-batches 2
```

3. Run full baseline training:

```bash
python train_resnet50_baseline.py --epochs 8 --batch-size 32
```

4. Optional: export penultimate-layer features for all `imgs/train` images:

```bash
python train_resnet50_baseline.py --epochs 8 --save-features outputs/resnet50_baseline/train_features.npz
```

### Grad-CAM Visualization

- Script: `gradcam_resnet50_baseline.py`
- Default checkpoint: `outputs/resnet50_baseline/best_resnet50.pt`
- Default output folder: `outputs/resnet50_baseline/gradcam`
- Output:
  - Grad-CAM overlay images
  - prediction summary: `gradcam_summary.json` (new summaries are appended on each run)

Generate a Grad-CAM overlay for one image:

```bash
python gradcam_resnet50_baseline.py imgs/train/c0/img_100026.jpg
```

Generate Grad-CAM overlays for a small batch of images from a folder:

```bash
python gradcam_resnet50_baseline.py --image-dir imgs/test --max-images 10
```

By default, Grad-CAM is generated for the predicted class. To visualize a specific class, pass a class name or index:

```bash
python gradcam_resnet50_baseline.py imgs/train/c0/img_100026.jpg --target-class c0
```

### Demo Prediction Video

- Script: `demo_video_resnet50_baseline.py`
- Purpose: create a short `.mp4` showing per-image predictions, confidence, and (when available) ground-truth labels inferred from parent class folders (`c0`-`c9`). It can also generate a second `.mp4` with Grad-CAM overlays for the same sampled frames.

Create a short demo from test images:

```bash
python demo_video_resnet50_baseline.py --image-dir imgs/test --max-images 20 --fps 2
```

Create a labeled demo from train images (shows GT vs prediction match):

```bash
python demo_video_resnet50_baseline.py --image-dir imgs/train --max-images 20 --fps 2 --output-video outputs/resnet50_baseline/demo_train_labeled.mp4
```

Create both the standard prediction demo and a second Grad-CAM overlay video:

```bash
python demo_video_resnet50_baseline.py --image-dir imgs/test --max-images 20 --fps 2 --output-video outputs/resnet50_baseline/demo_prediction.mp4 --gradcam-video outputs/resnet50_baseline/demo_prediction_gradcam.mp4
```

Create only the Grad-CAM overlay video without rewriting the standard demo:

```bash
python demo_video_resnet50_baseline.py --image-dir imgs/test --max-images 20 --fps 2 --gradcam-video outputs/resnet50_baseline/demo_prediction_gradcam.mp4
```

Create a labeled Grad-CAM demo from train images (shows GT vs prediction match plus overlay):

```bash
python demo_video_resnet50_baseline.py --image-dir imgs/train --max-images 20 --fps 2 --gradcam-video outputs/resnet50_baseline/demo_train_labeled_gradcam.mp4
```

## Model Recommendations

### Primary Model

- **ConvNeXt Tiny**
  - Best overall balance of ease of use, speed, and performance
  - Good main model for final experiments

### Strong Baseline

- **ResNet50**
  - Easy to train, debug, and explain
  - Good first baseline and well suited for Grad-CAM

### Optional Comparison

- **EfficientNet V2-S**
  - Strong accuracy candidate
  - More computationally demanding than the two models above

## Evaluation Plan

- Use subject-aware train/validation splits based on `subject` in `driver_imgs_list.csv`
- Report accuracy for binary and multiclass setups
- Compare full-image and crop-based models under the same split protocol
- Include qualitative Grad-CAM visualizations for correct and incorrect predictions
- Analyze common failure cases, including background bias and driver-specific overfitting

## Expected Deliverables

- A working still-image distracted driver classification prototype
- Quantitative comparison of:
  - full image vs driver-region crops
  - crop-based variants
  - binary vs multiclass classification
- Grad-CAM visualizations showing model attention
- Final course report and presentation video

## Positive Impact

If successful, this project could inform the design of low-cost driver monitoring systems that detect distraction early and support safer driving. More broadly, it explores how interpretable computer vision tools can contribute to reducing preventable accidents caused by inattention.

## Team

- Team name: **TBD: To Be Detected**
