# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 12:21:33 2025

@author: uig67136
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

#Load Preprocessed Data
X_train = np.load('C:/Users/uig67136/.spyder-py3/python scripts/RadarClassification/Data/Processed/X_train.npy')
X_test = np.load('C:/Users/uig67136/.spyder-py3/python scripts/RadarClassification/Data/Processed/X_test.npy')
Y_train = np.load('C:/Users/uig67136/.spyder-py3/python scripts/RadarClassification/Data/Processed/Y_train.npy')
Y_test = np.load('C:/Users/uig67136/.spyder-py3/python scripts/RadarClassification/Data/Processed/Y_test.npy')

#Initialize Random Forest Model
rf = RandomForestClassifier(
    n_estimators=100,   # number of trees
    max_depth=None,     # tree depth (None = expand fully)
    random_state=42,    # reproducibility
    n_jobs=-1           # use all CPU cores
)
#explain components
#n_estimatots= no. of decision trees
#random state= seed--same split and results in every run (same rows & cols(features) split)
#max-depth-how deep a tree can grow

#Train Model
rf.fit(X_train, Y_train)

#Prediction
Y_pred = rf.predict(X_test)

#Evaluation
acc = accuracy_score(Y_test, Y_pred)
print(f"Random Forest Accuracy: {acc:.2f}")

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred, target_names=['car','pedestrian','cyclist']))