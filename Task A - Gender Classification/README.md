# Task A: Gender Classification (Binary Classification)

## Overview

This repository contains the implementation for **Task A: Gender Classification** — distinguishing male vs female faces using deep learning.

🟡 Trained **from scratch** — no pretrained models used.
🟡 Achieves strong accuracy with deep CNN + strong data augmentation (Mixup, CutMix).

---

## Model Details

* **Architecture**: Deep CNN with 4 conv layers + BatchNorm + Dropout
* **Optimizer**: AdamW
* **Scheduler**: ReduceLROnPlateau
* **Data Augmentation**:

  * Random Horizontal Flip
  * Random Rotation
  * RandomResizedCrop
  * ColorJitter
  * **Mixup** and **CutMix**

---

## Training & Validation Results

| Metric    | Validation |
| --------- | ---------- |
| Accuracy  | 0.8839     |
| Precision | 0.9298     |
| Recall    | 0.9271     |
| F1-Score  | 0.9285     |

---

## Plots

See folder: `plots/`

* `loss.png` — Training loss per epoch
* `accuracy.png` — Train & Val accuracy per epoch

---

## Folder Structure

```
.
├── model.py
├── dataloader.py
├── config.py
├── train.py
├── test.py
├── predict.py
├── model/
│    └── best_gender_classifier.pth
├── plots/
│    ├── loss.png
│    └── accuracy.png
└── README.md
```

---

## Usage

### 🔑 Train Model

```bash
python train.py
```

Saves:

* Best model: `model/best_gender_classifier.pth`
* Plots: `plots/loss.png`, `plots/accuracy.png`

---

### 🔑 Test Model (on folder of images)

The test script accepts a **test data path** with same folder structure (`male/`, `female/`)
Returns: Accuracy, Precision, Recall, F1-score
Outputs: test_results.csv

```bash
python test.py <test_data_folder>
```

Example:

```bash
python test.py Task_A/test/
```

---

### 🔑 Predict Single Image

```bash
python predict.py <image_path>
```

Example:

```bash
python predict.py sample_image.png
```

Outputs:

```text
Prediction: male (98.5%)
```

---

## Pretrained Weights

The best model is saved to:

```text
model/best_gender_classifier.pth
```

You can load this and test new data.

---

## Submission Notes

🟡 Source code documented
🟡 Pretrained weights included
🟡 Test script works on folder input
🟡 Single image prediction also included
🟡 Results (metrics + plots) included

---