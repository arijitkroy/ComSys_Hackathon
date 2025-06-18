import os
import config
from dataset import FaceRecognitionDataset
from model import get_model
from train import train_model
from utils import set_seed
from torchvision import transforms

if __name__ == "__main__":
    set_seed(42)

    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    train_dataset = FaceRecognitionDataset(config.TRAIN_DIR, config.IMG_SIZE, transform)
    val_dataset = FaceRecognitionDataset(config.VAL_DIR, config.IMG_SIZE, transform)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    folder_names = sorted(os.listdir(config.TRAIN_DIR))
    num_classes = len(folder_names)
    print(f"Number of classes: {num_classes}")

    model = get_model(num_classes=num_classes, device=config.DEVICE)

    train_model(model, train_dataset, val_dataset, config)