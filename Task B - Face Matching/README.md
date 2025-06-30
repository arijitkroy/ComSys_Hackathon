## 🧠 Task B - Face Matching (Multi-Class Recognition with Distorted Inputs)

This project implements a **face verification system** that can match distorted or altered face images to their correct identities. Unlike classification, this task focuses on **verifying whether two face images belong to the same person**, even under challenging conditions.

---

## 📁 Dataset Structure

The dataset is organized as follows:

```
dataset/
├── 001_frontal/
│   ├── clean1.jpg
│   ├── clean2.jpg
│   └── distortion/
│       ├── distorted1.jpg
│       └── distorted2.jpg
├── 002_frontal/
│   ├── ...
...
```

* Each identity (e.g., `001_frontal/`) contains clean reference images.
* Distorted versions are inside a nested `distortion/` subfolder.

---

## 🎯 Objective

Build a **generalizable face embedding model** that:

* Brings similar (same identity) images close in embedding space.
* Pushes different identities far apart.
* Correctly verifies whether a distorted face belongs to a reference identity.

---

## 🛠️ Project Structure

```
.
├── config.py                # All training & path configs
├── train.py                 # Main training script
├── evaluate.py              # Computes Top-1 Accuracy & Macro F1
├── test.py                  # Command-line script to test custom input
├── siamese_model.py         # Siamese network & backbone model
├── utils.py                 # Embedding functions, contrastive loss, image loader
├── data_loader.py           # Dataset loader for face pairs
```

---

## 🏗️ Model Architecture

* **Backbone**: ResNet18 (ImageNet pretrained)
* **Network**: Siamese Network

  * Takes two images and compares embeddings
* **Loss**: Contrastive Loss
* **Metric**: Euclidean Distance

---

## ⚙️ Configuration (`config.py`)

```python
# Paths
TRAIN_DIR = "Task_B/train"
VAL_DIR = "Task_B/val"
MODEL_SAVE_PATH = "model/best_identity_model.pth"

# Image settings
IMG_SIZE = (224, 224)

# Training settings
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MARGIN = 1.0  # for contrastive loss

# Inference
THRESHOLD = 0.375  # Distance threshold for verification
DEVICE = "cuda"  # or "cpu"
```

---

## 🚀 Training

```bash
python train.py
```

* Trains using contrastive loss
* Shows **training accuracy live** using `tqdm`
* Saves the best model (`best_identity_model.pth`)
* Saves `accuracy_history.pt` and `f1_history.pt` for plotting

---

## 📊 Evaluation

```bash
python evaluate.py
```

* Compares distorted images to reference identities
* Outputs:

  * ✅ **Top-1 Accuracy**
  * ✅ **Macro-averaged F1 Score**
* Uses binary (0/1) match result for metric calculation

---

## 🧪 Test on Custom Folder

```bash
python test.py path/to/distorted_images/
```

* Runs face matching on custom folder of distorted images
* Outputs matched identity or "no match" in console

---

## 🧠 How Matching Works

1. **Embedding Generation**:

   * Both test and reference images are passed through the same embedding model.
2. **Distance Calculation**:

   * Uses Euclidean distance between embeddings.
3. **Thresholding**:

   * If distance < `THRESHOLD`, it’s a match.

---

## ✅ Evaluation Metrics

| Metric         | Value  |
| -------------- | ------ |
| Top-1 Accuracy | 0.8693 |
| Macro F1-Score | 0.8047 |

---

## 🏁 Future Improvements

* Use Triplet Loss or ArcFace for more robust embeddings.
* Add Face Alignment or MTCNN for preprocessing.
* Use ONNX or TorchScript for deployment.

---

## 🙌 Acknowledgements

* Based on ideas from [FaceNet](https://arxiv.org/abs/1503.03832) and Siamese Networks
* Pretrained models via `torchvision.models`

---