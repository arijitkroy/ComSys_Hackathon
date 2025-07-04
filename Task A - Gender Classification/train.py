# -------------------------------
# Training Script — Gender Classification (CNN)
# -------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from model import GenderCNN
from dataloader import get_data_loaders
import config
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from utils import mixup_data, mixup_criterion
from sklearn.metrics import f1_score

# -------------------------------
# Create directory for plots
# -------------------------------

os.makedirs('plots', exist_ok=True)

# -------------------------------
# Main Training Loop
# -------------------------------

if __name__ == "__main__":

    # --------------------
    # Setup device and model
    # --------------------

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    model = GenderCNN().to(device)

    # --------------------
    # Load DataLoaders
    # --------------------

    train_loader, val_loader = get_data_loaders()

    # --------------------
    # Calculate class weights from train dataset
    # --------------------

    train_dataset = train_loader.dataset  # ImageFolder
    targets = [sample[1] for sample in train_dataset.samples]
    class_counts = Counter(targets)

    # Sorted by class index (0 = female, 1 = male)
    class_counts_list = [class_counts[i] for i in range(len(train_dataset.classes))]

    print(f"Class counts: {class_counts_list}")

    total_samples = sum(class_counts_list)
    class_weights = torch.FloatTensor(
        [total_samples / c if c > 0 else 0.0 for c in class_counts_list]
    ).to(device)

    # --------------------
    # Loss, Optimizer, Scheduler
    # --------------------

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # --------------------
    # Tracking
    # --------------------

    best_val_acc = 0.0

    train_loss_list = []
    train_acc_list = []
    val_acc_list = []

    # --------------------
    # Training loop
    # --------------------

    for epoch in range(config.EPOCHS):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.EPOCHS}]", leave=True)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            # --------------------
            # Mixup Data Augmentation
            # --------------------
            aug_prob = random.random()

            if aug_prob < 0.7:
                mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2)
            else:
                mixed_images = images
                targets_a = targets_b = labels
                lam = 1

            # --------------------
            # Forward / Backward pass
            # --------------------
            optimizer.zero_grad()
            outputs = model(mixed_images)

            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # --------------------
            # Accuracy tracking
            # --------------------
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=running_loss/len(train_loader), acc=100*correct/total)

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # --------------------
        # Validation loop
        # --------------------

        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        val_loop = tqdm(val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100 * val_correct / val_total
        f1 = f1_score(all_labels, all_preds, average='binary')

        print(f"\n[Epoch {epoch+1}/{config.EPOCHS}] Train Acc: {train_acc:.2f}%  Val Acc: {val_acc:.2f}%  F1-score: {f1:.4f}\n\n")

        # --------------------
        # Save best model
        # --------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"--> Best model saved! Val Acc improved to {best_val_acc:.2f}%\n")

        # --------------------
        # Scheduler step
        # --------------------
        scheduler.step(val_acc)

        # --------------------
        # Log for plots
        # --------------------
        train_loss_list.append(avg_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

    # --------------------
    # Final plots
    # --------------------

    epochs = np.arange(1, config.EPOCHS + 1)

    # Loss plot
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_loss_list, label='Train Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/loss.png')

    # Accuracy plot
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_acc_list, label='Train Acc', color='blue')
    plt.plot(epochs, val_acc_list, label='Val Acc', color='green')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/accuracy.png')

    # --------------------
    # Final message
    # --------------------
    print("Training complete.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {config.MODEL_SAVE_PATH}")
    print("Plots saved to: plots/")