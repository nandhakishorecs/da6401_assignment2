import torch                    # type: ignore
import torch.nn as nn           # type: ignore
import torch.optim as optim     # type: ignore
from tqdm import tqdm           # type: ignore
import wandb                    # type: ignore

map_optimiser = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
}

map_loss = {
    'cross_entropy': nn.CrossEntropyLoss
}

map_activation = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'elu': nn.ELU,
    'mish': nn.Mish,
    'gelu': nn.GELU,
}

class ImageClassifier(nn.Module):
    def __init__(
            self,
            
            # Conv parameters
            input_size: tuple = (64, 64),
            n_layers: int = 5,
            in_channels: int = 3,
            n_classes: int = 10,
            kernel_size: int = 3,
            padding_mode: str = 'same',
            n_filters: int = 8,
            filter_strategy: str = 'same',
            bias: bool = True,
            
            # Classifier parameters
            n_epochs: int = 1,
            n_neurons: int = 128,
            activation: str = 'relu',
            optimiser: str = 'adam',
            criterion: str = 'cross_entropy',
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            batch_norm: bool = True,
            drop_out: float = 0.1,
            
            # Logging
            use_wandb: bool = False,
            name: str = 'DA24M011',
            validation: bool = True
    ) -> None:
        super(ImageClassifier, self).__init__()

        # CNN + Act + Pool
        self._n_layers = n_layers
        self._input_size = input_size
        self._in_channels = in_channels
        self._kernel_size = kernel_size
        self._bias = bias
        self._padding_mode = padding_mode.lower()
        self._filter_strategy = filter_strategy.lower()
        self._n_filters = n_filters
        self._activation = map_activation[activation.lower()]

        # Dense
        self._n_neurons = n_neurons

        # Classification
        self._n_epochs = n_epochs
        self._optimiser = map_optimiser[optimiser.lower()]
        self._learning_rate = learning_rate
        self._criterion = map_loss[criterion.lower()]()
        self._validation = validation

        # Regularisation
        self._weight_decay = weight_decay
        self._batch_norm = batch_norm
        self._drop_out = drop_out

        # For repr
        self._name = name
        self._loss_name = criterion
        self._optimiser_name = optimiser.lower()

        # wandb logging 
        self._log = use_wandb

        # Validate inputs
        if self._n_layers <= 0:
            raise ValueError('Number of CNN layers cannot be zero or negative')
        if self._kernel_size not in [3, 4, 5]:
            raise ValueError("kernel_size must be 3, 4, or 5")
        if self._filter_strategy not in ['same', 'double', 'half']:
            raise ValueError("filter_strategy must be 'same', 'double', or 'half'")
        if self._padding_mode not in ['same', 'valid']:
            raise ValueError("padding_mode must be 'same' or 'valid'")


        # Define filter progression
        channels = []
        filters = self._n_filters
        for i in range(self._n_layers):
            channels.append(filters)
            if self._filter_strategy == 'double':
                filters *= 2
                if filters > 8192:
                    filters = self._n_filters
            elif self._filter_strategy == 'half':
                filters /= 2
                filters = int(filters)
                if filters < 1:
                    filters = int(self._n_filters / 2)

        # Build convolutional blocks
        self.convolution_blocks = nn.ModuleList()
        current_channels = in_channels

        for out_channels in channels:
            # Calculate padding for 'same' mode
            if self._padding_mode == 'same':
                # TensorFlow 'SAME' padding: output_size = ceil(input_size / stride)
                # For stride=1, padding = (kernel_size - 1) / 2
                padding = (self._kernel_size - 1) // 2
            else:  # 'valid'
                padding = 0

            layers = [
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=self._kernel_size,
                    padding=padding, 
                    bias = self._bias
                )
            ]
            if self._batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(self._activation())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if(self._drop_out > 0): 
                layers.append(nn.Dropout(p = self._drop_out))
            
            self.convolution_blocks.append(nn.Sequential(*layers))
            current_channels = out_channels

        # Calculate size for dense layer
        conv_block_len = self._compute_output_size()
        if conv_block_len <= 0:
            raise ValueError('Receptive Field Collapse, check the model architecture')

        # Define dense layers
        self.dense_layer = nn.Linear(conv_block_len, self._n_neurons)
        if self._drop_out > 0.0:
            self.dropout_layer = nn.Dropout(p=self._drop_out)
        self.final_layer = nn.Linear(self._n_neurons, n_classes)

    # Calculate the number of features before the dense layer
    def _compute_output_size(self):
        dummy_input = torch.ones(1, self._in_channels, *self._input_size)
        self.eval()
        with torch.no_grad():
            x = dummy_input
            for _, block in enumerate(self.convolution_blocks):
                x = block(x)
                # Debug: Print shape after each block
                # print(f"Block {i} output shape: {x.shape}")
        
        if x.size(2) < 1 or x.size(3) < 1:
            raise ValueError(f"Feature map reduced to invalid size: {x.shape}. Increase input_size or adjust architecture.")
        
        return x.view(1, -1).size(1)

    def forward(self, X):
        for block in self.convolution_blocks:
            X = block(X)
        X = X.view(X.size(0), -1)
        X = self.dense_layer(X)
        X = self.final_layer(X)
        return X

    def fit(self, train_loader, val_loader=None, device: str = 'cpu'):
        # Convert device to torch.device if string
        if isinstance(device, str):
            device = torch.device(device)
        
        # Move model to device
        self.to(device)

        if (self._weight_decay > 0 and self._optimiser_name == 'adam'):
            optimizer = self._optimiser(self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay, betas = (0.99, 0.999))
        elif (self._weight_decay > 0 and self._optimiser_name == 'sgd'):
            optimizer = self._optimiser(self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay, momentum = 0.99)

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        # Debug: Print before starting loop
        # print(f"Starting training on device: {device}")
        # print(f"Number of training batches: {len(train_loader)}")
        if self._validation and val_loader is not None:
            # print(f"Number of validation batches: {len(val_loader)}")
            pass

        progress_bar = tqdm(
            range(self._n_epochs),
            unit="Epoch",
            ncols=100,
            dynamic_ncols=True
        )
        tqdm.write('\033[1;32mTraining\033[0m')

        for epoch in progress_bar:
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Debug: Print when entering batch loop
            # print(f"Epoch {epoch+1}: Processing training batches...")
            try:
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = self(images)
                    loss = self._criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # Debug: Print batch progress
                    if batch_idx % 10 == 0:
                        # print(f"Batch {batch_idx}/{len(train_loader)}")
                        pass

            except Exception as e:
                # print(f"Error in training loop: {str(e)}")
                raise

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            # Validation phase
            if self._validation and val_loader is not None:
                self.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = self(images)
                        loss = self._criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                val_loss = val_loss / len(val_loader)
                val_acc = 100 * correct / total
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
            else:
                val_loss = float('nan')
                val_acc = float('nan')
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

            # Log to WandB if enabled
            if (self._log):
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss if self._validation and val_loader is not None else None,
                    "val_accuracy": val_acc if self._validation and val_loader is not None else None
                })

            # Update progress bar
            if self._validation and val_loader is not None:
                progress_bar.set_postfix({
                    'Train Loss': f'\033[1;32m{train_loss:.2f}\033[0m',
                    'Train Acc': f'\033[1;32m{train_acc:.2f}\033[0m',
                    'Val Loss': f'\033[1;32m{val_loss:.2f}\033[0m',
                    'Val Acc': f'\033[1;32m{val_acc:.2f}\033[0m'
                })
            else:
                progress_bar.set_postfix({
                    'Train Loss': f'\033[1;32m{train_loss:.2f}\033[0m',
                    'Train Acc': f'\033[1;32m{train_acc:.2f}\033[0m'
                })

        if self._validation and val_loader is not None:
            return train_losses, val_losses, train_accuracies, val_accuracies
        return train_losses, train_accuracies

    def __repr__(self) -> str:
        return f'''\033[1;32mImage Classifier using CNNs\033[0m
    \033[1;32mModel Details:\033[0m
    \033[1;36mModel Name\033[0m                                     : {self._name}
    \033[1;36mNumber of CNN-Activation-Pooling blocks\033[0m        : {self._n_layers}
        \033[1;36mConvolution Kernel size\033[0m                        : {self._kernel_size}
        \033[1;36mNumber of Convolution Filters\033[0m                  : {self._n_filters}
        \033[1;36mFilter Changing Strategy\033[0m                       : {self._filter_strategy}
    \033[1;36mActivation\033[0m                                     : {self._activation.__name__}
    \033[1;36mOptimiser\033[0m                                      : {self._optimiser.__name__}
    \033[1;36mEpochs\033[0m                                         : {self._n_epochs}
    \033[1;36mLoss Function\033[0m                                  : {map_loss[self._loss_name.lower()].__name__}
    \033[1;36mNumber of Neurons in Dense Layer\033[0m               : {self._n_neurons}
    \033[1;36mLearning Rate\033[0m                                  : {self._learning_rate}
    \033[1;36mWeight Decay\033[0m                                   : {self._weight_decay}
    \033[1;36mDropout\033[0m                                        : {self._drop_out}
    \033[1;36mBatch Normalisation\033[0m                            : {self._batch_norm}
    \033[1;36mValidation\033[0m                                     : {self._validation}
    '''