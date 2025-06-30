import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

import config
from data_loader import FacePairDataset
from siamese_model import SiameseNet, EmbeddingNet
from utils import contrastive_loss, plot_metrics


def evaluate(model, val_loader):
    """
    Evaluates the model on the validation set using accuracy and macro F1 score.

    Args:
        model: Trained Siamese model.
        val_loader: DataLoader for validation pairs.

    Returns:
        Tuple[float, float]: (accuracy, macro_f1_score)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2 = img1.to(config.DEVICE), img2.to(config.DEVICE)
            emb1, emb2 = model(img1, img2)
            dists = torch.norm(emb1 - emb2, dim=1)
            preds = (dists < config.THRESHOLD).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1


def train_model():
    """
    Trains the Siamese network using contrastive loss.
    Tracks training loss and accuracy, evaluates on validation set after each epoch,
    and saves the best-performing model based on macro F1 score.
    """
    # Image transformations applied to both training and validation images
    transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor()
    ])

    # Load training and validation datasets
    dataset = FacePairDataset(config.TRAIN_DIR, transform)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    val_dataset = FacePairDataset(config.VAL_DIR, transform)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize Siamese model and optimizer
    model = SiameseNet(EmbeddingNet()).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_f1 = 0  # Track best F1 score for model saving
    accuracy_history = []
    f1_history = []

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        # Progress bar for training
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for img1, img2, label in progress:
            img1, img2, label = img1.to(config.DEVICE), img2.to(config.DEVICE), label.float().to(config.DEVICE)
            emb1, emb2 = model(img1, img2)
            loss = contrastive_loss(emb1, emb2, label, margin=config.MARGIN)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute training accuracy for current batch
            dists = torch.norm(emb1 - emb2, dim=1)
            preds = (dists < config.THRESHOLD).float()
            correct = (preds == label).sum().item()
            total_correct += correct
            total_samples += label.size(0)

            # Update progress bar with live metrics
            train_acc = total_correct / total_samples
            avg_loss = total_loss / (progress.n + 1)
            progress.set_postfix(loss=f"{avg_loss:.4f}", train_acc=f"{train_acc:.4f}")

        # Validation after each epoch
        avg_loss = total_loss / len(loader)
        val_acc, val_f1 = evaluate(model, val_loader)
        accuracy_history.append(val_acc)
        f1_history.append(val_f1)

        print(f"Validation ➤ Accuracy: {val_acc:.4f}, Macro F1: {val_f1:.4f}")

        # Save accuracy and F1 score history for plotting
        torch.save(accuracy_history, "plots/accuracy_history.pt")
        torch.save(f1_history, "plots/f1_history.pt")

        # Save model if F1 score improves
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"✅ Best model updated (Macro F1 = {best_f1:.4f})")

    print(f"\n📦 Best model saved to: {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    # Entry point for training and metric plotting
    train_model()
    plot_metrics()