import os
import torch
import matplotlib.pyplot as plt

def save_checkpoint(model, config, epoch):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(config.CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved to {save_path}")

def save_best_model(model, config):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Best model saved to {save_path}")

def set_seed(seed=42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_metrics(history, config):
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Plot Training Loss
    plt.figure(figsize=(8,6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epoch')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config.LOG_DIR, 'train_loss.png'))
    plt.close()

    # Plot Validation Accuracy
    plt.figure(figsize=(8,6))
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs Epoch')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config.LOG_DIR, 'val_accuracy.png'))
    plt.close()