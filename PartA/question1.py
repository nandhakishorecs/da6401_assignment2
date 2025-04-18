import torch                                # type: ignore
import torch.nn as nn                       # type: ignore
from torchvision import datasets            # type: ignore
from torch.utils.data import DataLoader     # type: ignore
from network import ImageClassifier
from data_loader import get_data_transforms

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
torch.manual_seed(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# Get data transforms
input_size = 64
train_transform, val_transform = get_data_transforms(input_size=input_size)

# Load custom dataset using ImageFolder
dataset_root = '/Users/nandhakishorecs/Documents/IITM/Jan_2025/DA6401/Assignments/Assignment2/PartA/dataset/inaturalist_12K'  # Update this path if needed
train_dataset = datasets.ImageFolder(
    root=f'{dataset_root}/train',
    transform=train_transform
)
val_dataset = datasets.ImageFolder(
    root=f'{dataset_root}/val',
    transform=val_transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Determine number of classes
n_classes = len(train_dataset.classes)

# Initialize model
model = ImageClassifier(
    input_size = (input_size, input_size),
    n_layers = 5,
    in_channels = 3,  # Adjust to 1 for grayscale images
    n_classes = n_classes,
    kernel_size = 4,  # Choose from [3, 4, 5]
    n_filters = 32,
    filter_strategy = 'half',  # Options: 'same', 'double', 'half'
    padding_mode = 'same',
    n_epochs = 2,
    n_neurons = 128,
    activation = 'relu',
    optimiser = 'sgd',
    criterion = 'cross_entropy',
    learning_rate = 1e-3,
    weight_decay = 1e-4,
    batch_norm = True,
    drop_out = 0.1,
    use_wandb = False,
    name = 'DA24M011',
    validation = True
).to(device)


print(model)

try:
    train_losses, val_losses, train_acc, val_acc = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
except Exception as e:
    print(f"Training failed with error: {str(e)}")
    raise

# # Print final results
# print(f"Final Training Accuracy: {train_acc[-1]:.2f}%")
# print(f"Final Validation Accuracy: {val_acc[-1]:.2f}%")