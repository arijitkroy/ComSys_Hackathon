# -------------------------------
# Configuration File - config.py
# -------------------------------

# Directory paths for training and validation datasets
TRAIN_DIR = "Task_B/train"
VAL_DIR = "Task_B/val"

# Path to save the best-performing Siamese model
MODEL_SAVE_PATH = "model/best_identity_model.pth"

# Input image dimensions (Height, Width) used for resizing
IMG_SIZE = (224, 224)

# Training hyperparameters
EPOCHS = 5                 # Number of training epochs
BATCH_SIZE = 32            # Number of image pairs per batch
LEARNING_RATE = 1e-4       # Learning rate for optimizer
MARGIN = 1.0               # Margin used in contrastive loss to separate dissimilar pairs

# Inference settings
THRESHOLD = 0.375            # Distance threshold below which a match is considered valid
DEVICE = "cuda"            # Device used for computation ("cuda" for GPU, "cpu" for CPU)