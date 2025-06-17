import torch
import torch.nn as nn
import torch.optim as optim
from model import GenderCNN
from dataloader import get_data_loaders
import config
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from utils import mixup_data, cutmix_data, mixup_criterion

# Create plot dir
os.makedirs('plots', exist_ok=True)

if __name__ == "__main__":
    # --- Setup ---
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    model = GenderCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer + Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Load data
    train_loader, val_loader = get_data_loaders()

    # --- Track best val acc ---
    best_val_acc = 0.0

    # Logs for plotting
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []

    # --- Training loop ---
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.EPOCHS}]", leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            # --- Randomly choose Mixup or CutMix ---
            aug_prob = random.random()
            if aug_prob < 0.5:
                mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.4)
            else:
                mixed_images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
            
            optimizer.zero_grad()
            outputs = model(mixed_images)
            
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # For tracking acc
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(loss=running_loss/len(train_loader), acc=100*correct/total)
        
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        
        val_loop = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        
        print(f"\n[Epoch {epoch+1}/{config.EPOCHS}] Train Acc: {train_acc:.2f}%  Val Acc: {val_acc:.2f}%\n")
        
        # --- Save best model ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"--> Best model saved! Val Acc improved to {best_val_acc:.2f}%\n")
        
        # --- LR scheduler step ---
        scheduler.step(val_acc)

        # Log for plot
        train_loss_list.append(avg_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

    # --- Final plots ---
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

    print("Training complete.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {config.MODEL_SAVE_PATH}")
    print("Plots saved to: plots/")