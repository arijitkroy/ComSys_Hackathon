import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FaceRecognitionDataset(Dataset):
    def __init__(self, root_dir, img_size, transform=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}

        self._prepare_dataset()

    def _prepare_dataset(self):
        folder_names = sorted(os.listdir(self.root_dir))
        for idx, folder in enumerate(folder_names):
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            self.label_map[folder] = idx
            all_images = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            # Include distortion images
            distortion_folder = os.path.join(folder_path, "distortion")
            if os.path.exists(distortion_folder):
                all_images += [
                    os.path.join(distortion_folder, f)
                    for f in os.listdir(distortion_folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

            for img_path in all_images:
                self.image_paths.append(img_path)
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label