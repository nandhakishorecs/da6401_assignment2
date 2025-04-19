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
    kernel_size = 5,  # Choose from [3, 4, 5]
    n_filters = 64,
    filter_strategy = 'same',  # Options: 'same', 'double', 'half'
    padding_mode = 'same',
    n_epochs = 80,
    n_neurons = 128,
    activation = 'gelu',
    optimiser = 'adam',
    criterion = 'cross_entropy',
    learning_rate = 0.00027,
    weight_decay = 0.00133,
    batch_norm = True,
    drop_out = 0.31396,
    use_wandb = True,
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

# Print final results
print(f"Final Training Accuracy: {train_acc[-1]:.2f}%")
print(f"Final Validation Accuracy: {val_acc[-1]:.2f}%")

# Save the trained model
model_path = 'model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# ---------- TESTING -----------------------

test_dataset = datasets.ImageFolder(
    root=f'{dataset_root}/test',
    transform=val_transform
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Load the saved model
loaded_model = ImageClassifier(
    n_layers = 5,
    in_channels = 3,  # Adjust to 1 for grayscale images
    n_classes = n_classes,
    kernel_size = 5,  # Choose from [3, 4, 5]
    n_filters = 64,
    filter_strategy = 'same',  # Options: 'same', 'double', 'half'
    padding_mode = 'same',
    n_epochs = 80,
    n_neurons = 128,
    activation = 'gelu',
    optimiser = 'adam',
    criterion = 'cross_entropy',
    learning_rate = 0.00027,
    weight_decay = 0.00133,
    batch_norm = True,
    drop_out = 0.31396,
    use_wandb = False,
    name = 'DA24M011',
    validation = True
).to(device)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()
print(f"Model loaded from {model_path}")

# Calculate test accuracy using the loaded model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images, labels
        predictions = loaded_model.predict(images)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"Final Test Accuracy: {test_accuracy:.2f}%")