import os
import sys
import torch
from tqdm import tqdm
from siamese_model import SiameseNet, EmbeddingNet
from utils import load_image, get_embedding, euclidean_dist
import config

# -----------------------
# Command-line Interface
# -----------------------

# Ensure a distorted folder path is passed as argument
if len(sys.argv) != 2:
    print("Usage: python test.py <path_to_distorted_folder>")
    sys.exit(1)

user_input_folder = sys.argv[1]

# Validate folder path
if not os.path.isdir(user_input_folder):
    print("Invalid folder path.")
    sys.exit(1)

# -----------------------
# Model Initialization
# -----------------------

# Load the trained Siamese model with embedding network
model = SiameseNet(EmbeddingNet()).to(config.DEVICE)
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
model.eval()

# -------------------------------
# Build Reference Embeddings
# -------------------------------

# For each identity in the training set, compute embeddings from clean images
reference_embeddings = {}
for identity in os.listdir(config.TRAIN_DIR):
    identity_path = os.path.join(config.TRAIN_DIR, identity)
    if not os.path.isdir(identity_path):
        continue
    for file in os.listdir(identity_path):
        if file == "distortion":
            continue  # Skip distorted folder
        img_path = os.path.join(identity_path, file)
        img_tensor = load_image(img_path).to(config.DEVICE)
        emb = get_embedding(model, img_tensor)
        reference_embeddings.setdefault(identity, []).append(emb)

# -------------------------------
# Predict for Distorted Inputs
# -------------------------------

print(f"\nMatching distorted images from: {user_input_folder}")
distorted_images = []
for identity in os.listdir(user_input_folder):
    distortion_dir = os.path.join(user_input_folder, identity, 'distortion')
    if os.path.exists(distortion_dir):
        for img in os.listdir(distortion_dir):
            distorted_images.append(os.path.join(distortion_dir, img))

for img_path in tqdm(distorted_images, desc="Matching"):
    if not os.path.isfile(img_path):
        continue

    distorted_tensor = load_image(img_path).to(config.DEVICE)
    distorted_emb = get_embedding(model, distorted_tensor)

    matched = False

    # Compare with all reference embeddings
    for identity, emb_list in reference_embeddings.items():
        for ref_emb in emb_list:
            dist = euclidean_dist(distorted_emb, ref_emb)
            if dist < config.THRESHOLD:
                print(f"{os.path.basename(img_path)} ➤ MATCHED with {identity} (distance = {dist:.4f})")
                matched = True
                break
        if matched:
            break

    # If no match is found under the threshold
    if not matched:
        print(f"{os.path.basename(img_path)} ➤ NO MATCH (all distances above threshold)")