import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import GenderCNN
import config
import sys
from tqdm import tqdm
import csv
import os

# Load model
device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
model = GenderCNN().to(device)
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
model.eval()

# Image transform
transform_test = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Class names
class_names = ['female', 'male']

# Custom Dataset that returns path
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        # Original tuple: (image, label)
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # Get image path
        path = self.samples[index][0]
        # Append path to tuple
        return original_tuple + (path,)

def evaluate_test_folder(test_folder):
    test_dataset = ImageFolderWithPaths(test_folder, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []
    csv_rows = []

    loop = tqdm(test_loader, desc="Testing", leave=True)
    with torch.no_grad():
        for images, labels, paths in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for path, pred in zip(paths, preds.cpu().numpy()):
                filename = os.path.basename(path)
                csv_rows.append([filename, class_names[pred]])

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print("\n--- Test Metrics ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    # Save CSV
    with open("test_results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "prediction"])
        writer.writerows(csv_rows)

    print("Predictions saved to test_results.csv")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <test_data_folder>")
    else:
        test_folder = sys.argv[1]
        evaluate_test_folder(test_folder)