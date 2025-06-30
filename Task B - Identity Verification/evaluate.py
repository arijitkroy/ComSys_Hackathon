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
    from each identity in the validation directory.

    Args:
        model: Trained Siamese embedding model.

    Returns:
        dict: {identity_name: [embedding1, embedding2, ...]}
    """
    reference_embeddings = {}
    for identity in os.listdir(config.VAL_DIR):
        identity_path = os.path.join(config.VAL_DIR, identity)
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

def predict_identity(emb, reference_embeddings):
    """
    Predicts the most similar identity for a given embedding by comparing it
    against all reference embeddings.

    Args:
        emb: Embedding of the distorted image.
        reference_embeddings: Dictionary of clean image embeddings.

    Returns:
        str or None: Predicted identity name if matched within threshold, else None.
    """
    best_match = None
    min_dist = float('inf')
    for identity, emb_list in reference_embeddings.items():
        for ref_emb in emb_list:
            dist = euclidean_dist(emb, ref_emb)
            if dist < min_dist:
                min_dist = dist
                best_match = identity
    return best_match if min_dist < config.THRESHOLD else None

def evaluate_model():
    """
    Loads a trained model and evaluates its face matching performance
    on distorted validation images using Top-1 Accuracy and Macro F1 Score.

    Returns:
        tuple: (y_true, y_pred) for further analysis or plotting.
    """
    # Load trained Siamese model
    model = SiameseNet(EmbeddingNet()).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()

    # Prepare embeddings of reference (clean) images
    reference_embeddings = build_reference_embeddings(model)

    y_true = []  # ground-truth labels (always 1 for distorted set)
    y_pred = []  # predicted binary match result (1 for correct match, else 0)

    for identity in os.listdir(config.VAL_DIR):
        distortion_dir = os.path.join(config.VAL_DIR, identity, 'distortion')
        if not os.path.exists(distortion_dir):
            continue

        for file in tqdm(os.listdir(distortion_dir), desc=f"Evaluating {identity}"):
            distorted_path = os.path.join(distortion_dir, file)
            distorted_tensor = load_image(distorted_path).to(config.DEVICE)
            distorted_emb = get_embedding(model, distorted_tensor)

            predicted_id = predict_identity(distorted_emb, reference_embeddings)

            # Label as 1 if correctly matched, else 0
            match = 1 if predicted_id == identity else 0
            y_true.append(1)         # Each distorted image is a known true identity
            y_pred.append(match)     # Model prediction: 1 (match), 0 (incorrect)

    # Compute evaluation metrics
    top1_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    print(f"\n📊 Evaluation Results:")
    print(f"Top-1 Accuracy     : {top1_acc:.4f}")
    print(f"Macro F1-Score     : {macro_f1:.4f}")

    return y_true, y_pred

# Run evaluation if executed directly
if __name__ == "__main__":
    y_true, y_pred = evaluate_model()