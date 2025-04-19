# Part A: Training from scratch

## Question 1 (5 Marks)

The file [network.py](https://github.com/nandhakishorecs/da6401_assignment2/blob/main/PartA/network.py) contains the code a custom class which has a 5 layer convolution neural network with a function to do forward propagation. <br>

The number of filters, kernel size, padding size, number of neurons in dense layer, learning rate, optimiser, epochs can be customised. The model can be initialised by:

```python
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
```
The model is coded to work on both CPU and GPU. To run a specific version of model (with specific hyper parameters) run the following file. 
```console
$ python3 question1.py
```

## Question 2 (15 Marks)

The iNaturalist dataset is a image based dataset with images of various sizes. A PyTorch __transforms__ module is used to resize and normalise the images and load it in the model. <br>

Using wandb's sweep feature, a sweep is done to do hyper parameter search. <br>

A yaml file with the hyper parameters is loaded to do the hyper parameter search. The yaml file looks like this: 
```yaml 
program: sweep.py
method: bayes
metric:
  name: val_acc
  goal: maximize
parameters:
  epochs:
    values: [10, 20]
  learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  .
  .
  .
  activation:
    values: ['relu', 'silu', 'gelu', 'mish']
  filter_strategy:
    values: ['same', 'double', 'half']
  dropout_rate:
    distribution: uniform
    min: 0.0
    max: 0.9
```
To do the sweep, update the global variables in the file [sweep.py](https://github.com/nandhakishorecs/da6401_assignment2/blob/main/PartA/sweep.py) to limit the number of experiments, device specific parameters and image size and run the file in CLI. 
```console
$ python3 sweep.py
```



