# -------------------------------
# Model Evaluation Script
# -------------------------------

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import GenderCNN
import config
from tqdm import tqdm

# --------------------
# Load trained model
# --------------------

# Set device (GPU or CPU)
device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')

# Initialize model
model = GenderCNN().to(device)

# Load best saved model weights
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
model.eval()

# --------------------
# Prepare validation DataLoader
# --------------------

transform_val = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load validation dataset
val_dataset = datasets.ImageFolder(f'{config.DATA_DIR}/val', transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# --------------------
# Evaluation loop
# --------------------

all_preds = []
all_labels = []

val_loop = tqdm(val_loader, desc="Evaluating", leave=True)

with torch.no_grad():
    for images, labels in val_loop:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        # Store predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --------------------
# Compute Evaluation Metrics
# --------------------

acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

# --------------------
# Print Results
# --------------------

print("\n--- Evaluation Metrics ---")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")