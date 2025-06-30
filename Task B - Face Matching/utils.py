import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import config
import matplotlib.pyplot as plt

def plot_metrics():
    """
    Loads accuracy and F1 score histories from disk and plots them over training epochs.
    Saves the plot as a PNG and displays it.
    """
    accuracy_history = torch.load("accuracy_history.pt")
    f1_history = torch.load("f1_history.pt")

    epochs = list(range(1, len(accuracy_history) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy_history, label="Top-1 Accuracy", marker='o', color='green')
    plt.plot(epochs, f1_history, label="Macro F1 Score", marker='s', color='purple')

    plt.title("Model Performance Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot/metrics_plot.png")
    plt.show()

def get_transform():
    """
    Returns the transformation pipeline applied to input face images.
    """
    return transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor()
    ])

def load_image(img_path):
    """
    Loads and preprocesses an image for inference.

    Args:
        img_path (str): Path to the input image.

    Returns:
        Tensor: Preprocessed image tensor of shape (1, C, H, W).
    """
    img = Image.open(img_path).convert("RGB")
    transform = get_transform()
    return transform(img).unsqueeze(0)  # Add batch dimension

def get_embedding(model, img_tensor):
    """
    Extracts the feature embedding for a single image using the model's embedding network.

    Args:
        model: Trained Siamese model.
        img_tensor: Input image tensor of shape (1, C, H, W).

    Returns:
        np.ndarray: Embedding vector.
    """
    model.eval()
    with torch.no_grad():
        emb = model.embedding_net(img_tensor.to(config.DEVICE)).cpu().numpy()
    return emb

def euclidean_dist(emb1, emb2):
    """
    Computes the Euclidean distance between two embeddings.

    Args:
        emb1, emb2 (np.ndarray): Embedding vectors.

    Returns:
        float: Euclidean distance.
    """
    return np.linalg.norm(emb1 - emb2)

def cosine_sim(emb1, emb2):
    """
    Computes cosine similarity between two embedding vectors.

    Args:
        emb1, emb2 (np.ndarray): Embedding vectors.

    Returns:
        float: Cosine similarity score.
    """
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return np.dot(emb1, emb2)

def contrastive_loss(emb1, emb2, label, margin=1.0):
    """
    Calculates contrastive loss between a pair of embeddings.

    Args:
        emb1, emb2 (Tensor): Embedding tensors.
        label (Tensor): 1 for similar, 0 for dissimilar pairs.
        margin (float): Margin for dissimilar pair penalty.

    Returns:
        Tensor: Computed contrastive loss.
    """
    dists = torch.norm(emb1 - emb2, dim=1)
    return torch.mean(label * dists**2 + (1 - label) * torch.clamp(margin - dists, min=0)**2)