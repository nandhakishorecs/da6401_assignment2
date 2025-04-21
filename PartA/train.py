import argparse
import wandb                                # type: ignore
import torch                                # type: ignore
import torch.nn as nn                       # type: ignore
from torchvision import datasets            # type: ignore
from torch.utils.data import DataLoader     # type: ignore
from network import ImageClassifier
from data_loader import get_data_transforms

import warnings
warnings.filterwarnings("ignore")
N_WORKERS = 2

def get_args():
    parser = argparse.ArgumentParser(description = '\033[92m' + '\nTrain a CNN based model with iNaturalist dataset\n' + '\033[0m')
    
    # Device
    parser.add_argument('-d', '--device', type = str, default = 'cpu', help = 'Use cuda if available')

    # Neural netwrok architecture 
    parser.add_argument('-sz', '--layer_size', type = int, default = 5, help = 'Number of Convolution layers')
    parser.add_argument('-ic', '--in_channels', type = int, default = 3, choices=[1, 3], help = 'Number of Input Channels')
    parser.add_argument('-is', '--in_size', type = int, default = 64, help = 'Size of Input image')
    parser.add_argument('-f', '--filter_size', type = int, default = 16, help = 'Number of Convolution filters')
    parser.add_argument('-ks', '--kernel_size', type = int, default = 3, choices = [3, 4, 5], help = 'Kernel size for Convolution filters')
    parser.add_argument('-fs', '--filter_strategy', type = str, default = 'same', choices = ['same', 'half', 'double'], help = 'Filter Strategy')
    parser.add_argument('-p', '--padding_mode', type = str, default = 'same', choices = ['same', 'valid'], help = 'Padding for images')
    parser.add_argument('-e', '--epochs', type = int, default = 1, help = 'Number of epochs')
    parser.add_argument('-bn', '--batch_norm', type = bool, default = True, choices = [True, False], help = 'Use of Batch Normalisation')
    parser.add_argument('-do', '--drop_out', type = float, default = 0.0,  help = 'Use of Drop Out')
    parser.add_argument('-dn', '--n_dense_neurons', type = int, default = 128,  help = 'Number of neurons in dense layer')
    
    # layer parameters 
    parser.add_argument('-b', '--bias', type = bool, default = True, help = 'Bias for convolution layer')
    parser.add_argument('-a', '--activation', type = str, default = 'relu', choices=['relu', 'gelu', 'mish', 'elu', 'silu'], help = 'Activation function for convolution layer')
    parser.add_argument('-bs', '--batch_size', type = int, default = 16, help = 'Batch size')
    parser.add_argument('-w_d', '--weight_decay', type = float, default = 0, help = 'Weight decay')
    
    # optimiser parameters
    parser.add_argument('-lr', '--learning_rate', type = float, default = 1e-4, help = 'Learning rate')
    parser.add_argument('-o', '--optimiser', type = str, default = 'adam', choices=['sgd', 'adam'], help = 'Optimiser')

    # model structure 
    parser.add_argument('-v', '--validation', type = bool, default = True, help = 'Use validation')
    
    # wandb configuration
    parser.add_argument('-log', '--log', type = bool, default = False, help = 'Use wandb')
    parser.add_argument('-wp', '--wandb_project', type = str, default = 'da6401_assignment2', help = 'Use wandb')
    parser.add_argument('-we', '--wand_entity', type = str, default = 'trial1', help = 'Use wandb')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # wandb initialisation
    if(args.log):
        wandb.init(
            project = args.wandb_project,  # project name 
            name = args.wand_entity
        )
    
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    # Get data transforms
    input_size = args.in_size
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=N_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=N_WORKERS)

    # Determine number of classes
    n_classes = len(train_dataset.classes)

    model = ImageClassifier(
        input_size = (input_size, input_size),
        n_layers = args.layer_size,
        in_channels = args.in_channels,  # Adjust to 1 for grayscale images
        n_classes = n_classes,
        kernel_size = args.kernel_size,  # Choose from [3, 4, 5]
        n_filters = args.filter_size,
        filter_strategy = args.filter_strategy,  # Options: 'same', 'double', 'half'
        padding_mode = args.padding_mode,
        n_epochs = args.epochs,
        n_neurons = args.n_dense_neurons,
        activation = args.activation,
        optimiser = args.optimiser,
        criterion = 'cross_entropy',
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        batch_norm = args.batch_norm,
        drop_out = args.drop_out,
        use_wandb = args.log,
        name = 'DA24M011',
        validation = args.validation
    ).to(args.device)

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
        input_size = (input_size, input_size),
        n_layers = args.layer_size,
        in_channels = args.in_channels,  # Adjust to 1 for grayscale images
        n_classes = n_classes,
        kernel_size = args.kernel_size,  # Choose from [3, 4, 5]
        n_filters = args.filter_size,
        filter_strategy = args.filter_strategy,  # Options: 'same', 'double', 'half'
        padding_mode = args.padding_mode,
        n_epochs = args.epochs,
        n_neurons = args.n_dense_neurons,
        activation = args.activation,
        optimiser = args.optimiser,
        criterion = 'cross_entropy',
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        batch_norm = args.batch_norm,
        drop_out = args.drop_out,
        use_wandb = args.log,
        name = 'DA24M011',
        validation = args.validation
    ).to(args.device)


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