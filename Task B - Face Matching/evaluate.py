import os
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from siamese_model import SiameseNet, EmbeddingNet
from utils import load_image, get_embedding, euclidean_dist
import config

def build_reference_embeddings(model):
    """
    Constructs a dictionary of embeddings for all reference (clean) images 
    from each identity in the training directory.

    Args:
        model: Trained Siamese embedding model.

    Returns:
        dict: {identity_name: [embedding1, embedding2, ...]}
    """
    reference_embeddings = {}
    for identity in os.listdir(config.TRAIN_DIR):
        identity_path = os.path.join(config.TRAIN_DIR, identity)
        if not os.path.isdir(identity_path):
            continue
        for file in os.listdir(identity_path):
            if file == "distortion":
                continue
            img_path = os.path.join(identity_path, file)
            img_tensor = load_image(img_path).to(config.DEVICE)
            emb = get_embedding(model, img_tensor)
            reference_embeddings.setdefault(identity, []).append(emb)
    return reference_embeddings

def is_match(distorted_emb, reference_embeddings):
    """
    Checks if the distorted embedding is close enough to any reference embedding
    based on a Euclidean distance threshold.

    Args:
        distorted_emb: Embedding vector of distorted image.
        reference_embeddings: Dictionary of clean embeddings from training identities.

    Returns:
        int: 1 if a match is found (distance < threshold), else 0.
    """
    for emb_list in reference_embeddings.values():
        for ref_emb in emb_list:
            dist = euclidean_dist(distorted_emb, ref_emb)
            if dist < config.THRESHOLD:
                return 1  # Positive match
    return 0  # No match

def evaluate_model():
    """
    Loads a trained model and evaluates its face matching performance
    using threshold-based binary verification against reference identities.

    Returns:
        tuple: (y_true, y_pred) for further analysis or plotting.
    """
    # Load trained Siamese model
    model = SiameseNet(EmbeddingNet()).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()

    # Prepare reference embeddings from clean training images
    reference_embeddings = build_reference_embeddings(model)

    y_true = []  # true = all distorted inputs should match someone
    y_pred = []  # model's decision (1 = match, 0 = no match)

    for identity in os.listdir(config.VAL_DIR):
        distortion_dir = os.path.join(config.VAL_DIR, identity, 'distortion')
        if not os.path.exists(distortion_dir):
            continue

        for file in tqdm(os.listdir(distortion_dir), desc=f"Evaluating {identity}"):
            distorted_path = os.path.join(distortion_dir, file)
            distorted_tensor = load_image(distorted_path).to(config.DEVICE)
            distorted_emb = get_embedding(model, distorted_tensor)

            match = is_match(distorted_emb, reference_embeddings)

            y_true.append(1)      # every distorted image has a true match
            y_pred.append(match)  # whether model matched it to anyone

    # Compute and print metrics
    top1_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    print(f"\n📊 Evaluation Results:")
    print(f"Top-1 Accuracy     : {top1_acc:.4f}")
    print(f"Macro F1-Score     : {macro_f1:.4f}")

    return y_true, y_pred

if __name__ == "__main__":
    y_true, y_pred = evaluate_model()