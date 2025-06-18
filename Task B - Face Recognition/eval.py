from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from tqdm import tqdm
import torch.nn.functional as F

def evaluate_model(model, val_loader, config):
    model.eval()

    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        loop = tqdm(val_loader, desc="Evaluating")

        for images, labels in loop:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            logits = model(images)

            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = correct / total
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\nValidation Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")

    return acc, precision, recall, f1