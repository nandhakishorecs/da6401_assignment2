import torch                                # type: ignore
from torchvision import datasets            # type: ignore
from torch.utils.data import DataLoader     # type: ignore
import wandb                                # type: ignore
import yaml                                 # type: ignore

from network import ImageClassifier
from data_loader import get_data_transforms

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
torch.manual_seed(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SWEEP_NAME = 'da6401_assignment2'
EXPERIMENT_COUNT = 120

with open("sweep_config.yml", "r") as file:
    sweep_config = yaml.safe_load(file)


sweep_id = wandb.sweep(sweep_config,project=SWEEP_NAME)

def do_sweep(): 
    wandb.init(project = 'da6401_assignment2')
    config = wandb.config 

    wandb.run.name = f'act_{config.activation}_lr_{config.learning_rate:.5f}_opt_{config.optimiser}_e_{config.epochs}_filter_{config.n_filters}_k_{config.kernel_size}_bias_{config.bias}_bs_{config.batch_size}_bn_{config.use_batch_norm}_da_{config.use_augmentation}_fs_{config.filter_strategy}_dropout_{config.dropout_rate:.5f}_dn_{config.n_dense_neurons}_wd_{config.weight_decay:.5f}'
    
    input_size = 128
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
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Determine number of classes
    n_classes = len(train_dataset.classes)

    # Initialize model
    model = ImageClassifier(
        input_size = (input_size, input_size),
        n_layers = 5,
        in_channels = 3,  # Adjust to 1 for grayscale images
        n_classes = n_classes,
        kernel_size = config.kernel_size,  # Choose from [3, 4, 5]
        n_filters = config.n_filters,
        filter_strategy = config.filter_strategy,  # Options: 'same', 'double', 'half'
        padding_mode = 'same',
        n_epochs = config.epochs,
        n_neurons = 128,
        activation = config.activation,
        optimiser = config.optimiser,
        criterion = 'cross_entropy',
        learning_rate = config.learning_rate,
        weight_decay = config.weight_decay,
        batch_norm = config.use_batch_norm,
        drop_out = config.dropout_rate,
        use_wandb = True,
        name = 'DA24M011',
        validation = True
    ).to(device)
    
    try:
        model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise

    wandb.finish()


if __name__ == '__main__': 
    wandb.agent(sweep_id, function = do_sweep, count = EXPERIMENT_COUNT)