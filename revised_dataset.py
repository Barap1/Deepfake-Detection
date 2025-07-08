# revised_dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed deepfake data from.npz files.
    
    This dataset is designed to work with data where each.npz file contains a
    'frames' array of shape (num_frames, height, width, channels) and a 'label' scalar.
    It applies on-the-fly augmentations for training.
    """
    def __init__(self, file_paths, labels, transform=None):
        """
        Args:
            file_paths (list): List of paths to the.npz files.
            labels (list): List of corresponding labels (0 for real, 1 for fake).
            transform (albumentations.Compose, optional): Augmentation pipeline.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Loads an.npz file, applies transformations, and returns the data.
        
        The.npz file is expected to contain a single array with the video frames.
        """
        try:
            with np.load(self.file_paths[idx]) as data:
                # CRITICAL FIX: Instead of hardcoding a key like 'frames' or 'arr_0',
                # dynamically get the first (and likely only) key from the.npz file.
                # This resolves the KeyError.
                if not data.files:
                    raise IOError(f"NPZ file is empty: {self.file_paths[idx]}")
                key = data.files[0]
                frames = data[key]
        except Exception as e:
            print(f"Error loading or processing file: {self.file_paths[idx]}")
            raise e

        # Apply augmentations to each frame individually
        if self.transform:
            augmented_frames = [self.transform(image=frame)['image'] for frame in frames]
            frames_tensor = torch.stack(augmented_frames)
        else:
            # This path is unlikely to be used with the current train script,
            # but is kept for completeness.
            frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return frames_tensor, label

def get_train_transforms(image_size=224):
    """
    Returns a composition of augmentations for the training set.
    These are "safe" augmentations that are unlikely to destroy deepfake artifacts.
    The transforms have been updated to use the modern albumentations API.
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=224):
    """
    Returns a composition of transformations for the validation/test set.
    No augmentations are applied, only resizing and normalization.
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=224):
    """
    Returns a composition of transformations for the validation/test set.
    No augmentations are applied, only resizing and normalization.
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])