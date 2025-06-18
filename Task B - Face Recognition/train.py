import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import save_checkpoint, save_best_model, plot_metrics

def train_model(model, train_dataset, val_dataset, config):
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_acc = 0.0
    history = {
        'train_loss': [],
        'val_acc': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.EPOCHS}]")

        for images, labels in loop:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

            loop.set_postfix(loss=loss.item(), acc=correct/total)

        epoch_loss = running_loss / len(train_loader)
        train_acc = correct / total

        history['train_loss'].append(epoch_loss)

        # Save checkpoint
        save_checkpoint(model, config, epoch)

        # Evaluate
        from eval import evaluate_model
        val_acc, precision, recall, f1 = evaluate_model(model, val_loader, config)

        history['val_acc'].append(val_acc)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)

        print(f"Epoch [{epoch+1}] Train_Acc: {train_acc:.4f} Val_Acc: {val_acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_best_model(model, config)

    # Plot after training
    plot_metrics(history, config)