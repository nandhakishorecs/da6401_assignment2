import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Model(nn.Module):
    def __init__(self, input_shape: tuple = (3, 64, 64), num_classes: int = 10, 
                 conv_filters: list = [32, 64, 128, 256, 512], 
                 kernel_sizes: list = [3, 3, 3, 3, 3],
                 activation_fn=nn.ReLU,
                 dense_neurons: int = 256):
        super(CNN_Model, self).__init__()

        # assert len(conv_filters) == 5, "You need exactly 5 conv layers"
        # assert len(kernel_sizes) == 5, "You need exactly 5 kernel sizes"

        self.conv_blocks = nn.Sequential()
        input_channels = input_shape[0]

        for i in range(5):
            self.conv_blocks.add_module(f'conv{i+1}', nn.Conv2d(input_channels, conv_filters[i], kernel_size=kernel_sizes[i], padding=kernel_sizes[i]//2))
            self.conv_blocks.add_module(f'act{i+1}', activation_fn())
            self.conv_blocks.add_module(f'maxpool{i+1}', nn.MaxPool2d(kernel_size=2, stride=2))
            input_channels = conv_filters[i]

        # Calculate final feature map size
        dummy_input = torch.zeros(1, *input_shape)
        with torch.no_grad():
            features = self.conv_blocks(dummy_input)
            flattened_size = features.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, dense_neurons),
            activation_fn(),
            nn.Linear(dense_neurons, num_classes)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)