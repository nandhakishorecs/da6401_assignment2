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