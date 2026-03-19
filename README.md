# TBD: To Be Detected

EECS 504 Computer Vision course project on still-image distracted driver detection using the State Farm Distracted Driver Detection dataset.
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/b649ed33-970b-4b4f-b12c-62e056e62512" />


## Overview

This project studies whether a vision model can detect distracted driving behavior from a single image and which visual regions matter most for that decision. We compare full-image classification against driver-centered crops and tighter crops around the face, hands, and phone. We also compare binary distracted-versus-safe classification with fine-grained multiclass classification and use Grad-CAM to verify whether the model attends to meaningful visual evidence.

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

