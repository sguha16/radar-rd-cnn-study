# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 21:30:04 2025

@author: uig67136
"""

import h5py
import numpy as np

# Open the HDF5 file
file_path = "E:/RadarProject/RadarScenes/RadarScenes/data/sequence_1/radar_data.h5"
with h5py.File(file_path, "r") as f:
    # List all datasets/groups in the file
    print("Keys:", list(f.keys()))
    
    # Suppose your dataset is named 'radar_data'
    data = f['radar_data'][:]  # load as NumPy array
# Check shape
print("Data shape:", data.shape)
print(list(data.dtype.names))  # list of column names

labels = data['label_id'][:]       # all labels
ranges = data['range_sc'][:]       # range per detection
velocities = data['vr'][:]         # radial velocity
rcs = data['rcs'][:]               # radar cross section
timestamp = data['timestamp'][:]   # timestamps

#bin to make range doppler maps for every few ms

#print("labels",labels)
# Extract labels from column 12 (index 11 in Python)
#labels = data[:, 11]  # numpy array of shape (num_samples,)
#print("Labels shape:", labels.shape)

# Optional: Extract features (all columns except label)
#features = np.delete(data, 11, axis=1)  # shape (num_samples, num_features)
#print("Features shape:", features.shape)