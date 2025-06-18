import torch
from torchvision import datasets, transforms
from torch.utils.data import WeightedRandomSampler, DataLoader
import config
import numpy as np

def get_data_loaders():
    transform_train = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=15),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    train_dataset = datasets.ImageFolder(f'{config.DATA_DIR}/train', transform=transform_train)
    val_dataset = datasets.ImageFolder(f'{config.DATA_DIR}/val', transform=transform_val)

    targets = train_dataset.targets  # for ImageFolder
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in targets]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, val_loader