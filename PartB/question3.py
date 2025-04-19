import torch                                            # type: ignore
import torch.nn as nn                                   # type: ignore
import torch.optim as optim                             # type: ignore
from torch.optim.lr_scheduler import StepLR             # type: ignore
import torchvision                                      # type: ignore
from torchvision import datasets, transforms, models    # type: ignore
from torch.utils.data import DataLoader                 # type: ignore
import copy                                             # type: ignore
from tqdm import tqdm                                   # type: ignore
import wandb                                            # type: ignore

# Initialize W&B
wandb.init(project="inaturalist_finetuning", config={
    "model": "ResNet50",
    "strategy": "freeze_layer1_layer2",
    "learning_rate": 0.001,
    "momentum": 0.9,
    "batch_size": 32,
    "epochs": 5,
    "dataset": "iNaturalist",
    "num_classes": 10,
    "scheduler": "StepLR",
    "step_size": 7,
    "gamma": 0.1
})

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load iNaturalist dataset (assuming folder structure: data/inaturalist/train and data/inaturalist/val)
data_dir = '"/Users/nandhakishorecs/Documents/IITM/Jan_2025/DA6401/Assignments/Assignment2/PartA/dataset/inaturalist_12K"'
image_datasets = {
    'train': datasets.ImageFolder(f'{data_dir}/train', data_transforms['train']),
    'val': datasets.ImageFolder(f'{data_dir}/val', data_transforms['val'])
}
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4)
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
num_classes = 10  # iNaturalist has 10 classes

# Function to set requires_grad for layers
def set_parameter_requires_grad(model, freeze_layers):
    for name, param in model.named_parameters():
        param.requires_grad = not freeze_layers(name)

# Create model with Strategy 2: Freeze layer1 and layer2
def get_model():
    model = models.resnet50(pretrained=True)
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Freeze layer1 and layer2, train layer3, layer4, and fc
    def freeze_layers(name): return any(x in name for x in ["layer1", "layer2"])
    set_parameter_requires_grad(model, freeze_layers)
    
    return model.to(device)

# Training function with tqdm progress bar and W&B logging
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Epoch progress bar
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # Create tqdm progress bar for the phase
            data_loader = dataloaders[phase]
            total_batches = len(data_loader)
            desc = f'{phase.capitalize()} Phase'
            progress_bar = tqdm(total=total_batches, desc=desc, leave=False)
            
            # Iterate over data
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data)
                running_loss += batch_loss
                running_corrects += batch_corrects
                
                # Update progress bar
                batch_size = inputs.size(0)
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': running_loss / ((progress_bar.n) * batch_size),
                    'acc': running_corrects.double() / ((progress_bar.n) * batch_size)
                })
            
            progress_bar.close()
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Log metrics to W&B
            wandb.log({
                f"{phase}_loss": epoch_loss,
                f"{phase}_accuracy": epoch_acc,
                "epoch": epoch + 1
            })
            
            # Print phase summary
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    print(f'Best val Acc: {best_acc:.4f}')
    # Log best validation accuracy to W&B
    wandb.log({"best_val_accuracy": best_acc})
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc

# Main execution
print("Training with Strategy 2: Freeze layer1 and layer2")
model = get_model()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.001, momentum=0.9
)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Train and evaluate
model, best_acc = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=5)

# Print final result
print(f"\nFinal Validation Accuracy: {best_acc:.4f}")

# Finish W&B run
wandb.finish()