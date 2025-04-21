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

## Question 3 (15 Marks)

Based on the above experiments, the following observations are made: 

The The iNaturalist dataset dataset has 10 classes of various species as images of different sizes. The number of samples per class is not same and the dataset has class imbalance. 

Based on the skewness in the class distribution, the contribution from the features of the samples in less numbers impacts the training of the model. 

For the given 5 layered model with CNN + Activation + MaxPool layer,  the following hyper parameters where found highly impactful: <br>
**Activation Function**: GELU <br>
**Convolution Kernel Size**: $5 \times 5$ <br>
**Convolution Padding Type**: 'Same' - Padding in such a way that the input size and output size remains the same. <br>
**Dropout**: Dropout probabilities from the range [0.3,0.45] gives better validation accuracy. <br>
**Batch Normalisation**: The sweeps without batch normalisation gave better validation accuracy.  <br>
**Data Augmentation**: The sweeps with Data Augmentation gave better validation accuracy. <br>
**Weight Decay**: Weight decay in the range [0.001,0.005] gives better validation accuracy. <br>
**Number of filters per Convolution-Activation-MaxPooling block**: Keeping the number of filters same as the initial number of filters gives better validation accuracy than doubling or halving the number of filters after each Convolution-Activation-MaxPooling block <br>
**Epochs**: Training the model from scratch with larger number of epochs gives better accuracy (40 epochs). 

## Question 4 (5 Marks) 

The Best Model chosen from  the sweeps: <br>
- Training Accuracy: 53.76 % <br>
- Validation Accuracy: 43.38 % <br>
- Activation Function: GELU <br>
- Learning Rate: 0.00027 <br>
- Optimiser: Adam<br>
- Epochs: 40 <br>
- Number of Convolution Filters: 64 <br>
- Convolution Kernel Size: $5 \times 5$ <br>
- Batch Size: 32 <br>
- Batch Normalisation: True <br>
- Data Augmentation: True <br>
- Filter Strategy: Same number of filters every Convolution-Activation-MaxPool block <br>
- Dropout: 0.31396 <br>
- Number of Neurons in Dense Layer: 128 <br>
- Weight Decay: 0.00133 <br>

Best Model accuracy on test set: 44.82% <br>
To get the 10 x 3 image grid, run the following file: 
```console
$ python3 grid.py
```

## Training the model

To train the model, use the train.py file with the following command line arguemnts

| Flag Name              | Default Value        | Description                                           |
|------------------------|----------------------|-------------------------------------------------------|
| -d, --device           | 'cpu'                | Use 'cuda' if GPU is available                        |
| -sz, --layer_size      | 5                    | Number of Convolution Layers                          |
| -ic, --in_channels     | 3                    | Number of Input Channels                              |
| -is, --in_size         | 64                   | Size of Input image                                   |
| -f, --filter_size      | 16                   | Number of Convolution Filters                         |
| -ks, --kernel_size     | 3                    | Convolutional filter size                             |
| -p, --padding_mode     | 'same'               | Padding strategy for inputs                           |
| -e, --epochs           | 1                    | Number of epochs for the model to train               |
| -bn, --batch_norm      | True                 | Use Batch Normalisation as regularistion              |
| -do, --drop_out        | 0.0                  | Use dropout regularisation, input dropout probability |
| -dn, --n_dense_neurons | 128                  | Number of neurons in the dense layer                  |
| -b, --bias             | True                 | Bias in Convolution layer                             |
| -a, --activation       | relu                 | Activation layer after convolutional layer            |
| -bs, --batch_size      | 64                   | Batch size for model training                         |
| -w_d, --weight_decay   | 0.0                  | Weight decay                                          |
| -lr, --learning_rate   | 0.001                | Learning Rate for the model                           |
| -o, --optimiser        | 'adam'               | Optimiser to minimise the model's loss                |
| -v, --validation       | True                 | Use Validation for training                           |
| -log, --log            | True                 | Use wandb logging                                     |
| -wp, --wandb_project   | 'da6401_assignment2' | Wandb project name                                    |
| -we, --wandb_entity    | 'trail1'             | Wandb entity name                                     |

The CLI arguments can be used by: 
```console
python3 train.py -d 'cpu' -sz 5 -ic 3 -is 224 -f 32 -ks 3 -fs 'same' -p 'same' -e 10 -bn True -do 0.1 -dn 128 -b True -a relu -bs 64 -w_d 0.001 -lr 0.001 -o 'adam' -v True -log True -wp 'da6401_assignment2' -we 'trail1'
```
