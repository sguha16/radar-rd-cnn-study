# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 09:12:14 2025

@author: uig67136
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import numpy as np
from CNN_model import RadarCNN  # import your CNN model
import matplotlib.pyplot as plt



def train_CNN(
    X_train, y_train, X_test, y_test,v_train,v_test,
    num_classes=3, num_epochs=5, batch_size_train=4, batch_size_test=6, lr=0.001
):
    """
    Train Radar CNN model.
    Args:
        x_train_path, y_train_path, x_test_path, y_test_path: paths to .npy files
        num_classes: number of output classes
        num_epochs: number of epochs
        batch_size_train: training batch size
        batch_size_test: test batch size
        lr: learning rate
    Returns:
        model: trained PyTorch model
    """



    # Reshape to (N,1,16,16)
    #X_train = np.reshape(X_train, (X_train.shape[0], 1, 16, 16))
    #X_test = np.reshape(X_test, (X_test.shape[0], 1, 16, 16))

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long).squeeze()
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long).squeeze()

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size_test)

    # =======================
    # 2. Model, Loss, Optimizer
    # =======================
    model = RadarCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # =======================
    # 3. Training Loop
    # =======================
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # =======================
    # 4. Evaluation
    # =======================
    all_preds = []
    all_labels = []
    model.eval()
    correctsum = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correctsum += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correctsum / total:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    #====checking RD plots after eval====
    # find one correct and one wrong sample
    correct_idx = []
    wrong_idx = []
    
    for i in range(len(all_labels)):
        if all_preds[i].item() == all_labels[i].item():
            correct_idx.append(i)
        else:
            wrong_idx.append(i)
        
    print("correct index",correct_idx)
    print("wrong index",wrong_idx)
    print("Number of wrong samples:", len(wrong_idx))
    print("shape of vtest array",np.shape(v_test))
    # extract RD maps (remove channel dim)
    rd_correct = X_test[correct_idx, 0]
    rd_wrong   = X_test[wrong_idx, 0]
    print("rd_correct shape",np.shape(rd_correct))
    print("rd_wrong shape",np.shape(rd_wrong))

    # plot side by side
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(rd_correct[0,:,:], aspect='auto')
    # plt.title(f"Correct\nTrue={y_test[correct_idx]}, Pred={predicted[correct_idx]}")
    plt.xlabel("Doppler")
    plt.ylabel("Range")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(rd_wrong[0,:,:], aspect='auto')
    # plt.title(f"Wrong\nTrue={y_test[wrong_idx]}, Pred={predicted[wrong_idx]}")
    plt.xlabel("Doppler")
    plt.ylabel("Range")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    #===correctness vs velocities
    correct_idx = np.array(correct_idx, dtype=int)
    wrong_idx   = np.array(wrong_idx, dtype=int)
    v_test = np.array(v_test)
    all_labels = np.array(all_labels)

    plt.figure(figsize=(6,4))
    plt.scatter(v_test[correct_idx],
            all_labels[correct_idx],
            c='g', label='Correct', alpha=0.6)

    plt.scatter(v_test[wrong_idx],
            all_labels[wrong_idx],
            c='r', label='Wrong', alpha=0.8)

    plt.xlabel("True velocity (m/s)")
    plt.ylabel("True class label")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("C:/Users/uig67136/.spyder-py3/python scripts/RadarClassification/Figures/velocity_vs_label_multitarget.png", dpi=150)
    plt.show()

    #====================================
    return model

