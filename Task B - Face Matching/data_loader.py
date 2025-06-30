import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FacePairDataset(Dataset):
    """
    Custom Dataset for generating image pairs (positive and negative) 
    for Siamese Network training in a face verification task.
    """
    def __init__(self, root_dir, transform=None):
        """
        Initializes the dataset by listing identity folders and generating training pairs.

        Args:
            root_dir (str): Root directory containing identity subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.identity_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
        self.samples = self._generate_pairs()  # List of (img1_path, img2_path, label) tuples

    def _generate_pairs(self):
        """
        Generates image pairs: positive (same identity) and negative (different identities).

        Returns:
            list: List of tuples containing (distorted_image, reference_image, label).
        """
        pairs = []
        for identity_path in self.identity_folders:
            # Collect reference (clean) images
            clean_imgs = [os.path.join(identity_path, f) for f in os.listdir(identity_path) if f != 'distortion']
            
            # Collect distorted images from 'distortion/' subfolder
            distortion_dir = os.path.join(identity_path, 'distortion')
            distorted_imgs = [os.path.join(distortion_dir, f) for f in os.listdir(distortion_dir)]

            # Generate positive pairs: (distorted, clean) from same identity
            for dimg in distorted_imgs:
                for cimg in clean_imgs:
                    pairs.append((dimg, cimg, 1))

            # Generate negative pairs: (distorted, clean) from different identities
            for _ in range(len(distorted_imgs)):
                other_identity = random.choice([p for p in self.identity_folders if p != identity_path])
                neg_img = random.choice([
                    os.path.join(other_identity, f) for f in os.listdir(other_identity) if f != 'distortion'
                ])
                dimg = random.choice(distorted_imgs)
                pairs.append((dimg, neg_img, 0))

        return pairs

    def __len__(self):
        """
        Returns:
            int: Total number of image pairs in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves an image pair and label by index.

        Args:
            idx (int): Index of the image pair.

        Returns:
            tuple: Transformed (image1, image2, label)
        """
        img1_path, img2_path, label = self.samples[idx]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label