# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 09:05:46 2025

@author: uig67136
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#Define CNN model
class RadarCNN(nn.Module):
    def __init__(self, num_classes=3):  # example: car, pedestrian, cyclist-3 logits for classification
        super(RadarCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)#batch normalizationafter convolution layer conv1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.LazyLinear(64)#no need to calc input features, output is 64
        #batch normalization after fully connected layer fc1
        self.bn_fc = nn.BatchNorm1d(64)
        #dropout
        self.dropout = nn.Dropout(0.5)  
        # 50% chance to drop Randomly "drops" (sets to 0) some activations during training.
        #Forces the network to not rely on specific neurons → reduces overfitting.
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Convolution + ReLU + Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)   # flatten all except batch--(batch,:)
        # Lazy initialization: fix fc1 input size at runtime
        #if self.fc1.in_features == 1:
           #self.fc1 = nn.Linear(x.size(1), 64).to(x.device) 
           #x.size(1) gives the actual size of second dimension
           #64 is the output dimension of the fully connected layer--> 3 labels
           #increase 64 to 128 for more features
        # Fully connected layers
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x