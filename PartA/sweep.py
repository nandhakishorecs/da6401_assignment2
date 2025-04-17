import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
import os
import sys
from PIL import Image

# Optimize for Tensor Cores on NVIDIA A30
torch.set_float32_matmul_precision('medium')

# Check wandb and PyTorch versions
try:
    print(f"wandb version: {wandb.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    if not hasattr(wandb, 'sweep'):
        print("Error: wandb.sweep is not available. Please upgrade wandb with 'pip install --upgrade wandb'.")
        sys.exit(1)
except AttributeError:
    print("Error: Unable to verify wandb version. Ensure wandb is installed correctly.")
    sys.exit(1)

# Custom dataset to handle invalid images
class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.invalid_files = []

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target
        except Exception as e:
            print(f"Skipping invalid image: {path} ({e})")
            self.invalid_files.append(path)
            dummy = torch.zeros(3, 224, 224) if self.transform else Image.new('RGB', (224, 224))
            return dummy, target

# CNN Classifier
class CNNCLassifier(pl.LightningModule):
    def __init__(
            self,
            n_layers: int,
            n_clases: int,
            input_channels: int,
            n_filters: int,
            kernel_size: int,
            padding_type: str = 'same',
            filter_strategy: str = 'same',
            use_batch_norm: bool = False,
            dropout_rate: float = 0.0,
            stride: int = 1,
            bias: bool = False,
            pool_size: int = 2,
            pool_stride: int = 1,
            pool_padding: int = 0,
            activation: str = 'relu',
            optimiser: str = 'adam',
            epochs: int = 10,
            batch_size: int = 128,
            learning_rate: float = 1e-3,
            accum_grad_batches: int = 2
    ):
        super(CNNCLassifier, self).__init__()
        self.save_hyperparameters(ignore=[
            'n_layers', 'n_filters', 'kernel_size', 'filter_strategy',
            'use_batch_norm', 'dropout_rate', 'optimiser', 'batch_size', 'learning_rate',
            'activation', 'bias', 'use_augmentation', 'padding_type'
        ])

        self._n_layers = n_layers
        self._n_classes = n_clases
        self._input_channels = input_channels
        self._n_filters = n_filters
        self._conv_kernel_size = kernel_size
        self._padding_type = padding_type.lower()
        self._filter_strategy = filter_strategy
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._conv_stride = stride
        self._b = bias
        self._pool_size = pool_size
        self._pool_stride = pool_stride
        self._pool_padding = pool_padding
        self._lr = learning_rate
        self._n_epochs = epochs
        self._batch_size = batch_size
        self._optimiser_name = optimiser.lower()
        self._accum_grad_batches = accum_grad_batches

        self._activation = {
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }.get(activation.lower(), nn.ReLU(inplace=True))

        self.conv_layers = nn.ModuleList()
        channels = input_channels
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(self._n_layers):
            if self._filter_strategy == 'same':
                filters = self._n_filters
            elif self._filter_strategy == 'double':
                filters = self._n_filters * (2 ** i)
            elif self._filter_strategy == 'half':
                filters = self._n_filters // (2 ** i) if i > 0 else self._n_filters
            else:
                filters = self._n_filters

            filters = max(filters, 1)

            # Compute padding for PyTorch
            padding = (self._conv_kernel_size - 1) // 2 if self._padding_type == 'same' else 0

            conv_block = []
            conv_block.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=filters,
                    kernel_size=self._conv_kernel_size,
                    padding=padding,
                    stride=self._conv_stride,
                    bias=self._b,
                    device=device
                )
            )
            if self._use_batch_norm:
                conv_block.append(nn.BatchNorm2d(filters, device=device))
            conv_block.append(
                nn.MaxPool2d(
                    kernel_size=self._pool_size,
                    stride=self._pool_stride,
                    padding=self._pool_padding
                )
            )
            conv_block.append(self._activation)
            
            self.conv_layers.append(nn.Sequential(*conv_block))
            channels = filters

        self.dropout = nn.Dropout(p=self._dropout_rate)
        self.dense_layer = nn.Linear(channels, 128, device=device)
        self.dense_activation = self._activation
        self.fc_layer = nn.Linear(128, self._n_classes, device=device)

    def forward(self, x):
        for i, conv_block in enumerate(self.conv_layers):
            x = conv_block(x)
            # print(f"Layer {i} output shape: {x.shape}")
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # print(f"After adaptive pool shape: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"After flatten shape: {x.shape}")
        x = self.dropout(x)
        x = self.dense_layer(x)
        # print(f"After dense layer shape: {x.shape}")
        x = self.dense_activation(x)
        x = self.fc_layer(x)
        # print(f"After fc layer shape: {x.shape}")
        return x

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y_true)
        
        predictions = torch.argmax(y_pred, dim=1)
        TP = (y_true == predictions).sum().item()
        total_predictions = y_true.size(0)

        self._train_TP += TP
        self._total_train_predictions += total_predictions
        self._train_accuracy = self._train_TP / self._total_train_predictions if self._total_train_predictions > 0 else 0.0
        self._train_loss = loss.item()

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', self._train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y_true)
        
        predictions = torch.argmax(y_pred, dim=1)
        TP = (y_true == predictions).sum().item()
        total_predictions = y_true.size(0)

        self._val_TP += TP
        self._total_val_predictions += total_predictions
        self._val_accuracy = self._val_TP / self._total_val_predictions if self._total_val_predictions > 0 else 0.0
        self._val_loss = loss.item()

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', self._val_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y_true)
        
        predictions = torch.argmax(y_pred, dim=1)
        TP = (y_true == predictions).sum().item()
        total_predictions = y_true.size(0)

        self._test_TP += TP
        self._total_test_predictions += total_predictions
        self._test_accuracy = self._test_TP / self._total_test_predictions if self._total_test_predictions > 0 else 0.0

        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_acc', self._test_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self):
        wandb.log({
            'epoch': self.current_epoch,
            'train_loss': self._train_loss,
            'train_acc': self._train_accuracy
        })

    def on_validation_epoch_end(self):
        wandb.log({
            'epoch': self.current_epoch,
            'val_loss': self._val_loss,
            'val_acc': self._val_accuracy
        })

    def configure_optimizers(self):
        optimizers = {
            'sgd': torch.optim.SGD(self.parameters(), lr=self._lr, momentum=0.9),
            'rmsprop': torch.optim.RMSprop(self.parameters(), lr=self._lr),
            'adam': torch.optim.Adam(self.parameters(), lr=self._lr)
        }
        return optimizers.get(self._optimiser_name, torch.optim.Adam(self.parameters(), lr=self._lr))
    
    def on_train_epoch_start(self):
        self._train_TP = 0
        self._total_train_predictions = 0
        self._train_accuracy = 0.0
    
    def on_validation_epoch_start(self):
        self._val_TP = 0
        self._total_val_predictions = 0
        self._val_accuracy = 0.0
    
    def on_test_epoch_start(self):
        self._test_TP = 0
        self._total_test_predictions = 0
        self._test_accuracy = 0.0

# Training function for wandb sweep
def train():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with wandb.init() as run:
        config = run.config
        
        data_dir = '/home/ee20d064/GAIL_Machine_Learning/Nandhakishore/CNN/PartA/dataset/inaturalist_12K'
        train_dir = f'{data_dir}/train'
        val_dir = f'{data_dir}/val'
        test_dir = f'{data_dir}/test'

        try:
            # mean, std = compute_dataset_stats(train_dir, batch_size=32)
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
        except Exception as e:
            print(f"Failed to compute dataset stats: {e}")
            run.log({'error': str(e)})
            return

        def get_train_transforms(enable_augmentation=True):
            transform_list = [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
            if enable_augmentation:
                transform_list.insert(1, transforms.RandomHorizontalFlip())
                transform_list.insert(2, transforms.RandomRotation(10))
            return transforms.Compose(transform_list)

        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transforms = val_transforms

        try:
            train_dataset = SafeImageFolder(root=train_dir, transform=get_train_transforms(config.use_augmentation))
            val_dataset = SafeImageFolder(root=val_dir, transform=val_transforms)
            test_dataset = SafeImageFolder(root=test_dir, transform=test_transforms)
        except Exception as e:
            print(f"Error loading datasets: {e}")
            run.log({'error': str(e)})
            return

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        try:
            model = CNNCLassifier(
                n_layers=config.n_layers,
                n_clases=10,
                input_channels=3,
                n_filters=config.n_filters,
                kernel_size=config.kernel_size,
                padding_type=config.padding_type,
                filter_strategy=config.filter_strategy,
                use_batch_norm=config.use_batch_norm,
                dropout_rate=config.dropout_rate,
                stride=1,
                bias=config.bias,
                pool_size=2,
                pool_stride=1,
                pool_padding=0,
                activation=config.activation,
                optimiser=config.optimiser,
                epochs=5,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                accum_grad_batches=2
            )

            csv_logger = CSVLogger(save_dir='logs', name='inaturalist_sweep')
            wandb_logger = pl.loggers.WandbLogger(project='inaturalist_sweep', log_model=False)

            trainer = pl.Trainer(
                max_epochs=model._n_epochs,
                callbacks=[TQDMProgressBar()],
                logger=[csv_logger, wandb_logger],
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                accumulate_grad_batches=model._accum_grad_batches
            )

            trainer.fit(model, train_dataloader, val_dataloader)
            trainer.test(model, test_dataloader)

            wandb.log({
                'final_train_acc': model._train_accuracy,
                'final_val_acc': model._val_accuracy,
                'final_test_acc': model._test_accuracy
            })

            if train_dataset.invalid_files or val_dataset.invalid_files or test_dataset.invalid_files:
                wandb.log({
                    'train_invalid_files': train_dataset.invalid_files,
                    'val_invalid_files': val_dataset.invalid_files,
                    'test_invalid_files': test_dataset.invalid_files
                })

        except RuntimeError as e:
            print(f"Run failed: {e}")
            run.log({'error': str(e)})
            return
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Main script
if __name__ == "__main__":
    try:
        with open('sweep_config.yaml', 'r') as file:
            sweep_config = yaml.safe_load(file)
        
        sweep_id = wandb.sweep(sweep_config, project='inaturalist_sweep')
        
        wandb.agent(sweep_id, function=train, count=1)
    except Exception as e:
        print(f"Error running sweep: {e}")
        print("Ensure wandb, PyTorch, and datasets are configured correctly.")
        sys.exit(1)