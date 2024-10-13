
---
layout: post
title: "Plant Disease Detection with Machine Learning"
date: 2024-10-14
categories: machine-learning, plant-disease
---

## Introduction

In this blog post, we'll explore how to build a machine learning model to classify plant diseases. With the increasing demand for sustainable agriculture, the ability to quickly identify plant diseases can make a significant difference for farmers and researchers alike.

## The Model

We used a convolutional neural network (CNN) to classify different types of plant diseases. Here’s a brief overview of the model architecture:

```python
import torch
import torch.nn as nn

# Neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, n_labels):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 1024)  # Adjust based on input size
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, n_labels)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu6(self.fc1(x)))
        x = self.fc2(x)
        return x

Preprocessing the Data

Before we train our model, we need to preprocess the images. This involves converting them to grayscale, resizing, and normalizing them. Here’s how you can do it:

python

from torchvision import transforms

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((256, 256)),  # Resize to expected input size
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_tensor

Making Predictions

Once the model is trained, you can make predictions on new images. Here’s a quick function to predict the class of an image:

python

def predict_image_class(image, model, label_dict):
    input_batch = preprocess_image(image)
    with torch.no_grad():
        output = model(input_batch)
        _, predicted = torch.max(output, 1)
    class_name = label_dict[predicted.item()]
    return class_name

Conclusion

By leveraging deep learning, we can create robust models to help identify plant diseases effectively. This approach not only benefits farmers but also contributes to food security by ensuring healthier crops.

Feel free to reach out if you have any questions or if you’d like to discuss this further. Happy coding! 🌱---
layout: post
title: "Plant Disease Detection with Machine Learning"
date: 2024-10-14
categories: machine-learning, plant-disease
---

## Introduction

In this blog post, we'll explore how to build a machine learning model to classify plant diseases. With the increasing demand for sustainable agriculture, the ability to quickly identify plant diseases can make a significant difference for farmers and researchers alike.

## The Model

We used a convolutional neural network (CNN) to classify different types of plant diseases. Here’s a brief overview of the model architecture:

```python
import torch
import torch.nn as nn

# Neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, n_labels):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 1024)  # Adjust based on input size
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, n_labels)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu6(self.fc1(x)))
        x = self.fc2(x)
        return x

Preprocessing the Data

Before we train our model, we need to preprocess the images. This involves converting them to grayscale, resizing, and normalizing them. Here’s how you can do it:

python

from torchvision import transforms

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((256, 256)),  # Resize to expected input size
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_tensor

Making Predictions

Once the model is trained, you can make predictions on new images. Here’s a quick function to predict the class of an image:

python

def predict_image_class(image, model, label_dict):
    input_batch = preprocess_image(image)
    with torch.no_grad():
        output = model(input_batch)
        _, predicted = torch.max(output, 1)
    class_name = label_dict[predicted.item()]
    return class_name

Conclusion

By leveraging deep learning, we can create robust models to help identify plant diseases effectively. This approach not only benefits farmers but also contributes to food security by ensuring healthier crops.

Feel free to reach out if you have any questions or if you’d like to discuss this further. Happy coding! 🌱
