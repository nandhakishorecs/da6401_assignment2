# For local testing, before wandb configuration 
import torch                                # type: ignore
import torch.nn as nn                       # type: ignore
from tqdm import tqdm                       # type: ignore
import matplotlib.pyplot as plt             # type: ignore
import torchvision.transforms as transforms # type: ignore

def get_data_transforms(input_size=64, in_channels=3, augmentation = True):
    # Define normalization based on in_channels
    if in_channels == 3:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif in_channels == 1:
        # Normalization for grayscale images
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        raise ValueError("in_channels must be 1 (grayscale) or 3 (RGB)")

    if(augmentation): 
    # Training transformations with data augmentation
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
    else: 
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize
        ])


    # Validation transformations (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize
    ])

    return train_transform, val_transform