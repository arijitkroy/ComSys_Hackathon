# -------------------------------
# Gender Classification CNN Model (4 Conv Layers)
# -------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class GenderCNN(nn.Module):
    def __init__(self):
        super(GenderCNN, self).__init__()

        # --------------------
        # Convolutional Layers
        # --------------------
        
        # Conv Block 1: 3 → 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv Block 2: 32 → 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv Block 3: 64 → 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Conv Block 4: 128 → 256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # --------------------
        # Pooling and Dropout
        # --------------------
        
        # MaxPooling: 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

        # --------------------
        # Fully Connected Layers
        # --------------------

        # Compute input size to first FC layer
        fc_input_dim = (
            256 * (config.IMG_SIZE // (2**4)) * (config.IMG_SIZE // (2**4))
        )  # 4 pooling layers → divide H,W by 2^4

        # FC1: → 512 units
        self.fc1 = nn.Linear(fc_input_dim, 512)

        # FC2: → 2 output classes (male / female)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        # Forward pass through conv layers with ReLU + BatchNorm + Pooling

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x