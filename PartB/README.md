### Part B - Fine Tuning a pre-exting model 

#### Model Choosen: ResNet50 

## Question 1 (5 Marks)

Loading the pre-trained model as it is and changing the classes to 10 for iNaturalist dataset. 

```python3 
# Load pre-trained ResNet50
model = models.resnet50(pretrained=True)

# Modify the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 10 classes for iNaturalist
```

## Question 2 (5 Marks) - Fine Tuning Strategies 

- The CNN + Feature extraction layers are freezed and the weights corresponding to these layers are downloaded from the internet.

- Keeping the CNN layers, adding more fully connected layers with same number of neurons and training the model 

- Freeze the first 'k' layers and training the remaining layers. 

## Question 3 (10 Marks) - Fine Tuning Resnet50 Model 

Chosen Strategy: Keep the feature extraction / CNN layer intact and remove the dense layer from the pre-trained model. A new dense layer with 10 outputs is added. 

The newly added dense layer is then trained from scratch. This is achieved through a user defined Python function. 

```python
def set_parameter_requires_grad(model, freeze_layers):
    for name, param in model.named_parameters():
        param.requires_grad = not freeze_layers(name)
def get_model():
    model = models.resnet50(pretrained=True)
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    
    # Freeze layer1 and layer2, train layer3, layer4, and fc
    def freeze_layers(name): return any(x in name for x in ["layer1", "layer2"])
    set_parameter_requires_grad(model, freeze_layers)
    
    return model.to(device)

```