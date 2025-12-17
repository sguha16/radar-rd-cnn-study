# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:37:51 2025

@author: uig67136
"""

from SynthData import generate_dataset
from preprocess_data import preprocess_data
from train_CNN import train_CNN

# =======================
# 1. Generate Synthetic Data
# =======================
# Adjust number of samples, etc. as needed
print("Step 1: Generating synthetic dataset...")
X,y,true_velocities=generate_dataset(N_samples=500)
print("Synthetic dataset generated.")
print("X shape:", X.shape)
print("y:", y)

#1. Load Data

# =======================
# 2. Preprocess Data
# =======================
print("Step 2: Preprocessing dataset...")
X_train, X_test, Y_train, Y_test, v_train, v_test=preprocess_data(X,y,true_velocities)
print("Preprocessing complete.")

# =======================
# 3. Train CNN
# =======================
print("Step 3: Training CNN...")
model = train_CNN(
    X_train,Y_train,X_test,Y_test,v_train, v_test,
    num_classes=3,  # Adjust to your labels
    num_epochs=15
)
print("Training complete.")

#can save optionally in Raw & Processed