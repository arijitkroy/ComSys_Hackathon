# -------------------------------
# Mixup Utilities
# -------------------------------

import torch
import numpy as np

# -------------------------------
# Mixup data generation
# -------------------------------
# Inputs:
#   x     → batch of input images (tensor)
#   y     → batch of labels (tensor)
#   alpha → mixup parameter (float), higher alpha → more mixed
# Returns:
#   mixed_x → mixed input images
#   y_a, y_b → target pairs for mixup loss
#   lam     → mixup ratio
# -------------------------------

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Mix images
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Prepare target pairs
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

# -------------------------------
# Mixup criterion (loss calculation)
# -------------------------------
# Combines two losses weighted by lam
# -------------------------------

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)