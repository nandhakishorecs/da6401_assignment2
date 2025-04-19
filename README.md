# DA6401 Introduction to Deep Learning - Assignment 2
This repository contains all the code for Assignment 2 for the Introduction to Deep Learning Course (DA6401) offered at Wadhwani School of Data Science and AI, Indian Institute of Technology Madras. 

**Course Instructor**: Prof. Mitesh Khapra <br>
**Author**: Nandhakishore C S <br>
**Roll Number**: DA24M011 

This Assignment has three sections (denoted by Part A, B & C) - refer the respective directories to check the code. 

## Problem Statement
**Graded**: In Part A and Part B of this assignment you will build and experiment with CNN based image classifiers using a subset of the [iNaturalist dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip) <br>


## Instructions for run the code: 

1. Create a virtual environment in Python and install the necessary libraries mentioned in the requirements.txt file. 
```console 
$ python3 -m venv env_name 
$ source env_name/bin/activate 
(env_name) $ pip install -r requirements.txt
```

2. The Following packages are needed to use the code in this repository. 
```txt
numpy 
torch 
torchvision
tqdm 
requests
wandb
PyYAML 
pillow
```
3. PyTorch is used to implement CNNs and to create DataLoaders for dataset. 
4. The metrics from different models are logged using wandb. Login to wandb by creating a new project and pasting the API key in CLI. 
```console 
$ wandb login 
```
