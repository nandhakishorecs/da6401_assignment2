import torch                                            # type: ignore
from torchvision import datasets                        # type: ignore
from torch.utils.data import DataLoader, Subset         # type: ignore
import wandb                                            # type: ignore
import random
import numpy as np                                      # type: ignore
from PIL import Image, ImageDraw, ImageFont             # type: ignore

from network import ImageClassifier
from data_loader import get_data_transforms

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize WandB
wandb.init(project="inaturalist_test_grid", name="test_predictions_grid_concise")

# Get data transforms
input_size = 128
_, val_transform = get_data_transforms(input_size=input_size)

# Load test dataset
dataset_root = '/home/ee20d064/GAIL_Machine_Learning/Nandhakishore/CNN/dataset/inaturalist_12K'
test_dataset = datasets.ImageFolder(
    root=f'{dataset_root}/test',
    transform=val_transform
)

# Create test data loader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Determine number of classes
n_classes = len(test_dataset.classes)
class_names = test_dataset.classes

# Initialize and load the model
model = ImageClassifier(
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

# Load the saved model
model_path = 'model.pth'
try:
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    raise

# Select 30 random test images
num_images = 30
indices = random.sample(range(len(test_dataset)), num_images)
subset_dataset = Subset(test_dataset, indices)
subset_loader = DataLoader(subset_dataset, batch_size=1, shuffle=False)

# Create WandB table for 10x3 grid
table = wandb.Table(columns=["Image 1", "Image 2", "Image 3"])

# Collect predictions and annotate images
images, true_labels, pred_labels = [], [], []
try:
    font = ImageFont.truetype("arial.ttf", 8)  # Adjust font size as needed
except:
    font = ImageFont.load_default()  # Fallback to default font

with torch.no_grad():
    for image, label in subset_loader:
        image, label = image.to(device), label.to(device)
        predicted = model.predict(image, device=device)
        
        # Convert image back to PIL
        img = image.cpu().squeeze(0).permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # Denormalize (assuming mean=0.5, std=0.5)
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        
        # Annotate image with predicted and true labels
        draw = ImageDraw.Draw(pil_img)
        pred_label = class_names[predicted.item()]
        true_label = class_names[label.item()]
        is_correct = predicted.item() == label.item()
        pred_color = (0, 255, 0) if is_correct else (255, 0, 0)  # Green for correct, red for incorrect
        
        # Draw text on image (top-left for pred, bottom-left for true)
        draw.text((5, 5), f"Pred: {pred_label}", fill=pred_color, font=font)
        draw.text((5, pil_img.height - 15), f"True: {true_label}", fill=(255, 255, 255), font=font)
        
        images.append(pil_img)
        true_labels.append(label.item())
        pred_labels.append(predicted.item())

# Populate the 10x3 grid
for row in range(10):
    row_data = []
    for col in range(3):
        idx = row * 3 + col
        if idx < len(images):
            row_data.append(wandb.Image(images[idx]))
        else:
            row_data.append(None)  # Empty cell if fewer than 30 images
    table.add_data(*row_data)

# Log the table to WandB
wandb.log({"Test Predictions Grid": table})

# Calculate overall test accuracy
correct = sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
test_accuracy = 100 * correct / num_images
wandb.log({"Test Accuracy (30 samples)": test_accuracy})
print(f"Test Accuracy on 30 samples: {test_accuracy:.2f}%")

# Finish WandB run
wandb.finish()