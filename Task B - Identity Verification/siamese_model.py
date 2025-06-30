import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class EmbeddingNet(nn.Module):
    """
    Embedding network that uses a pretrained ResNet18 backbone to 
    extract 128-dimensional feature embeddings from input face images.
    """
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet18 and remove the classification head (avgpool and fc)
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # Keep all layers except the last FC
        self.fc = nn.Linear(512, 128)  # Project 512D feature to 128D embedding space

    def forward(self, x):
        """
        Forward pass through the embedding network.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W)

        Returns:
            Tensor: 128-dimensional embedding vector
        """
        x = self.backbone(x).squeeze()  # Shape: (B, 512)
        return self.fc(x)               # Shape: (B, 128)

class SiameseNet(nn.Module):
    """
    Siamese network that processes two input images through a shared embedding network 
    and returns their corresponding embeddings.
    """
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        """
        Forward pass through the Siamese network.

        Args:
            x1 (Tensor): First input image batch
            x2 (Tensor): Second input image batch

        Returns:
            Tuple[Tensor, Tensor]: Embeddings for x1 and x2
        """
        emb1 = self.embedding_net(x1)
        emb2 = self.embedding_net(x2)
        return emb1, emb2