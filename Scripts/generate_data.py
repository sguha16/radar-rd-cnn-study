# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 11:13:21 2025

@author: uig67136
"""


#Step 1: Simulate radar data

#Goal: Create a small dataset that looks like radar signals, with labels for “targets.”

#Decide dataset size

#Example: 30 samples, 3 classes (car, pedestrian, cyclist)
#Each sample can be a small 1D array or a 2D “range-Doppler map” (e.g., 16×16 numbers)
#Generate features and labels
#Features: random numbers simulating signal intensities
#Labels: integers 0, 1, 2 for the 3 classes
#Keep it in your repo
#Save in radar_project/data/processed/dummy_data.npy for easy reuse

import numpy as np

# Parameters
num_samples = 30        # total samples
num_classes = 3         # car, pedestrian, cyclist
array_shape = (16, 16)  # 2D range-Doppler map size

# Initialize arrays
X = np.zeros((num_samples, *array_shape)) #30x16x16
y = np.zeros(num_samples, dtype=int)#30 1D array

# Generate dummy data
for i in range(num_samples): # i from 0 to 30 
    label = i % num_classes            # cycle through 0,1,2
    y[i] = label
    # random values simulating radar signal, slightly different per class
    X[i] = np.random.rand(*array_shape) + label * 0.5  #each sample has a 16*16 random array + 0.5*label


print("Dummy radar dataset created:")
print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Classes:", np.unique(y))

#Save label and samples 
np.save('C:/Users/uig67136/.spyder-py3/python scripts/RadarClassification/Data/Raw/dummy_X.npy', X)#samples
np.save('C:/Users/uig67136/.spyder-py3/python scripts/RadarClassification/Data/Raw/dummy_Y.npy', y)#labels