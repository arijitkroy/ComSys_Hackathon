# COMSYS Hackathon-5, 2025: Robust Face Recognition and Gender Classification under Adverse Visual Conditions

This repository contains two deep learning tasks:

* 🧠 **Task A: Gender Classification** – Classifies face images as male or female.
* 🧠 **Task B: Face Matching with Distorted Inputs** – Verifies whether a distorted face matches a clean identity.


## 🗂️ Overview of Tasks

### 🟩 Task A: Gender Classification (Binary Classification)

* Trained from scratch using a deep CNN.
* Incorporates advanced augmentation like **Mixup**.
* Evaluated on precision, recall, and F1-score.

### 🟦 Task B: Distorted Identity Verification (Face Matching)

* Uses a **Siamese Network** to verify if a distorted image belongs to a person.
* Based on **ResNet18** embeddings and **Contrastive Loss**.
* Focused on **face verification**, not classification.

---

## 📁 Dataset Format

### Task A: Gender Classification

```
Task_A/
├── train/
│   ├── male/
│   └── female/
├── val/
├── test/
```

### Task B: Face Matching

```
Task_B/
├── 001_frontal/
│   ├── clean1.jpg
│   └── distortion/
│       ├── distorted1.jpg
├── 002_frontal/
│   └── ...
```

---

## 🛠️ Project Structure

```
.
├── config.py                # Shared config
├── train.py                 # Task-specific training
├── test.py                  # Test on a folder (Task A/B)
├── evaluate.py              # Metric evaluation
├── predict.py               # Task A single image prediction
├── model.py / siamese_model.py
├── utils.py
├── data_loader.py
├── model/                   # Saved weights
├── plots/                   # Training curves
```

---

## 🏗️ Model Architectures

### Task A

* CNN with 4 conv blocks + BatchNorm + Dropout
* Optimizer: **AdamW**, Scheduler: **ReduceLROnPlateau**

### Task B

* Backbone: **ResNet18 (pretrained)**
* Siamese structure: Embeds two images and compares
* Loss: **Contrastive Loss**
* Distance: **Euclidean**

---

## 📈 Evaluation Metrics

| Task | Metric         | Value  |
| ---- | -------------- | ------ |
| A    | Accuracy       | 0.9147 |
| A    | Macro F1 Score | 0.9434 |
| A    | Precision      | 0.9404 |
| A    | Recall         | 0.9464 |
| B    | Top-1 Accuracy | 0.8510 |
| B    | Macro F1 Score | 0.7429 |

---

## 🚀 How to Run

### 🔧 Install Requirements

```bash
pip install -r requirements.txt
```

---

### 🟩 Task A: Gender Classification

#### ✅ Train

```bash
python train.py
```

#### ✅ Test

```bash
python test.py Task_A/test/
```

#### ✅ Predict Single Image

```bash
python predict.py path/to/image.jpg
```

---

### 🟦 Task B: Distorted Identity Verification

#### ✅ Train Siamese Model

```bash
python train.py
```

#### ✅ Evaluate

```bash
python evaluate.py
```

#### ✅ Test on Custom Distorted Folder

```bash
python test.py Task_B/sample_distorted/
```

---

## 📊 Visual Outputs

* `plots/loss.png`: Training loss curve (Task A)
* `plots/accuracy.png`: Accuracy curves (Task A)
* `plot/metrics_plot.png`: Accuracy & F1 (Task B)

---

## 🧪 How Face Matching Works (Task B)

1. Embedding generation for clean and distorted images
2. Distance calculation using Euclidean distance
3. Threshold-based decision (e.g., `< 0.7` = match)

---

## 🧠 Future Enhancements

* Switch to **Triplet Loss / ArcFace** for stronger face discrimination
* Use **MTCNN** or **Dlib** for automatic face alignment
* Deploy models using ONNX / TorchScript

---

## 🙌 Credits

* Inspired by [FaceNet](https://arxiv.org/abs/1503.03832), Siamese Networks, and deep classification techniques
* Trained using **PyTorch** and **torchvision**

---
