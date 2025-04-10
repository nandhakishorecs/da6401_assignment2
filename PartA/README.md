# Part A: Training from scratch

## Question 1 (5 Marks)

The file [network.py](<link>) contains the code a custom class which has a 5 layer convolution neural network with a function to do forward propagation. <br>

The number of filters, kernel size, pooling size, padding size and stride can be customised. The model can be initialised by:

```python
model = CNN_Model(input_shape = (3, 64, 64), 
                  num_classes = 10, 
                  conv_filters = [32, 64, 128, 256, 512], 
                  kernel_sizes = [3, 3, 3, 3, 3], 
                  activation_fn = nn.ReLU, 
                  dense_neurons = 256
)
```

## Question 2 (15 Marks)

The iNaturalist dataset is a image based dataset with 


