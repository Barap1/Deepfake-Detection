# revised_dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepfakeDataset(Dataset):
    
    def __init__(self, file_paths, labels, transform=None):

        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        try:
            with np.load(self.file_paths[idx]) as data:
 
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

            frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return frames_tensor, label

def get_train_transforms(image_size=224):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=224):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=224):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])