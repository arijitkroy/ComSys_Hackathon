# -------------------------------
# Dataloader with WeightedRandomSampler
# -------------------------------

import torch
from torchvision import datasets, transforms
from torch.utils.data import WeightedRandomSampler, DataLoader
import config
import numpy as np

# Function to return train_loader and val_loader
def get_data_loaders():
    
    # --------------------
    # Data Augmentation / Preprocessing
    # --------------------
    
    transform_train = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),        
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # --------------------
    # Dataset: ImageFolder format
    # --------------------
    
    train_dataset = datasets.ImageFolder(f'{config.DATA_DIR}/train', transform=transform_train)
    val_dataset = datasets.ImageFolder(f'{config.DATA_DIR}/val', transform=transform_val)

    # --------------------
    # WeightedRandomSampler to handle imbalance
    # --------------------
    
    targets = train_dataset.targets  # class labels for each sample
    class_counts = np.bincount(targets)  # count of each class
    class_weights = 1. / class_counts    # inverse of class count
    sample_weights = [class_weights[t] for t in targets]  # weight for each sample

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # --------------------
    # DataLoaders
    # --------------------
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        sampler=sampler, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, val_loader