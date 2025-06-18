# -------------------------------
# Single Image Gender Prediction Script
# -------------------------------

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import GenderCNN
import config
import sys

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
# Class Names
# --------------------

# Adjust these if your dataset folder names are different
class_names = ['female', 'male']

# --------------------
# Image Transform (same as training)
# --------------------

transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --------------------
# Prediction function
# --------------------

def predict_image(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, 1)

    # Print result
    print(f"Prediction: {class_names[pred_class.item()]} ({confidence.item() * 100:.2f}%)")

# --------------------
# Command-line interface
# --------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_image(sys.argv[1])