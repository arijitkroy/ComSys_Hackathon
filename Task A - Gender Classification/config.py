# -------------------------------
# Configuration Settings
# -------------------------------

# Path to main data directory (should contain 'train', 'val', 'test' subfolders)
DATA_DIR = 'Task_A'

# Batch size for training and validation
BATCH_SIZE = 32

# Total number of training epochs
EPOCHS = 20

# Initial learning rate for optimizer
LEARNING_RATE = 0.0005

# Input image size (images will be resized to IMG_SIZE x IMG_SIZE)
IMG_SIZE = 128

# Device to use: 'cuda' for GPU, 'cpu' for CPU
DEVICE = 'cuda'

# Path to save the best trained model (based on F1-Score)
MODEL_SAVE_PATH = 'model/best_gender_classifier.pth'