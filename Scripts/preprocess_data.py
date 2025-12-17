# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 11:46:52 2025

@author: uig67136
"""
# preprocess_data.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(X, Y, true_velocities, test_size=0.2, random_state=42):
    """
    Preprocesses radar dataset: scales features and splits into train/test sets.

    Args:
        X: np.ndarray of shape (N, 1, 16, 16) or (N, features)
        Y: np.ndarray of shape (N,)
        test_size: float, fraction of test data
        random_state: int, seed for reproducibility

    Returns:
        X_train, X_test, Y_train, Y_test
    """

    
    X_scaled=X/(np.max(X)+0.000001)
    #constant for safety if some samples are empty to avoid divide by 0

    # Train/test split
    X_train, X_test, Y_train, Y_test, v_train,v_test = train_test_split(
        X_scaled, Y, true_velocities,test_size=test_size, random_state=random_state, stratify=Y
    )

    print("✅ Preprocessing complete.")
    print(f"Training set: {X_train.shape}, {Y_train.shape}")
    print(f"Test set: {X_test.shape}, {Y_test.shape}")
    
    return X_train, X_test, Y_train, Y_test,v_train,v_test
