import torch                                    # type: ignore
import torch.nn as nn                           # type: ignore
import torch.optim as optim                     # type: ignore
import torchvision                              # type: ignore
import torchvision.transforms as transforms     # type: ignore
import torchvision.models as models             # type: ignore
from torch.utils.data import DataLoader         # type: ignore
from tqdm import tqdm                           # type: ignore
import os
import wandb                                    # type: ignore

# Initialize W&B
wandb.init(project="inaturalist_finetuning", config={
    "model": "ResNet50",
    "dataset": "iNaturalist_12K",
    "learning_rate": 0.001,
    "momentum": 0.9,
    "batch_size": 32,
    "epochs": 5,
    "num_classes": 10,
    "optimizer": "SGD",
    "data_dir": "/Users/nandhakishorecs/Documents/IITM/Jan_2025/DA6401/Assignments/Assignment2/PartA/dataset/inaturalist_12K"
})

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ImageNet input
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std
])

# Load iNaturalist dataset
data_dir = '/Users/nandhakishorecs/Documents/IITM/Jan_2025/DA6401/Assignments/Assignment2/PartA/dataset/inaturalist_12K'  # Update with actual path
train_dataset = torchvision.datasets.ImageFolder(
    root=os.path.join(data_dir, "train"),
    transform=transform
)
val_dataset = torchvision.datasets.ImageFolder(
    root=os.path.join(data_dir, "val"),
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load pre-trained ResNet50
model = models.resnet50(pretrained=True)

# Modify the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 10 classes for iNaturalist

# Move model to device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training function with tqdm and W&B logging
def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # Initialize tqdm progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        batch_loss = running_loss / (progress_bar.n + 1)
        batch_acc = 100 * correct / total
        progress_bar.set_postfix({"Loss": f"{batch_loss:.4f}", "Acc": f"{batch_acc:.2f}%"})
    
    progress_bar.close()
    
    # Log training metrics to W&B
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    wandb.log({
        "train_loss": epoch_loss,
        "train_accuracy": epoch_acc,
        "epoch": epoch + 1
    })
    
    return epoch_loss, epoch_acc

# Validation function with tqdm and W&B logging
def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    # Initialize tqdm progress bar
    progress_bar = tqdm(val_loader, desc="Validation", leave=True)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            batch_loss = running_loss / (progress_bar.n + 1)
            batch_acc = 100 * correct / total
            progress_bar.set_postfix({"Loss": f"{batch_loss:.4f}", "Acc": f"{batch_acc:.2f}%"})
    
    progress_bar.close()
    
    # Log validation metrics to W&B
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    wandb.log({
        "val_loss": epoch_loss,
        "val_accuracy": epoch_acc,
        "epoch": epoch + 1
    })
    
    return epoch_loss, epoch_acc

# Main execution
if __name__ == "__main__":
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
        val_loss, val_acc = validate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save the fine-tuned model (commented out as in original)
    # torch.save(model.state_dict(), "finetuned_resnet50.pth")
    # print("Model saved as finetuned_resnet50.pth")
    
    # Log final validation accuracy to W&B
    wandb.log({"final_val_accuracy": val_acc})
    
    # Finish W&B run
    wandb.finish()