# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 10:06:51 2025

@author: uig67136
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import numpy as np
import timm  # NEW: import timm
import matplotlib.pyplot as plt


def train_ViT(
    X_train, y_train, X_test, y_test, v_train, v_test,
    num_classes=3, num_epochs=50, batch_size_train=32, batch_size_test=32, lr=1e-4
):
    """
    Train Vision Transformer model for radar classification.
    """
    
    # ============ NEW: Resize data to square ============
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    # Resize from (N, 1, 64, 32) to (N, 1, 64, 64)
    X_train = F.interpolate(X_train, size=(64, 64), mode='bilinear', align_corners=False)
    X_test = F.interpolate(X_test, size=(64, 64), mode='bilinear', align_corners=False)
    # ====================================================
    
    # Convert labels to tensors
    y_train = torch.tensor(y_train, dtype=torch.long).squeeze()
    y_test = torch.tensor(y_test, dtype=torch.long).squeeze()

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size_test)

    # ============ NEW: Load ViT model ============
    model = timm.create_model(
        'vit_tiny_patch16_224',
        pretrained=False,
        num_classes=num_classes,
        img_size=64,
        in_chans=1  # grayscale
    )
    # =============================================
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # =======================
    # Training Loop (SAME AS BEFORE)
    # =======================
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # move to device
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # =======================
    # Evaluation (SAME AS BEFORE)
    # =======================
    all_preds = []
    all_labels = []
    model.eval()
    correctsum = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correctsum += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correctsum / total:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # =======================
    # Visualization (SAME AS BEFORE)
    # =======================
    correct_idx = []
    wrong_idx = []
    
    for i in range(len(all_labels)):
        if all_preds[i] == all_labels[i]:
            correct_idx.append(i)
        else:
            wrong_idx.append(i)
        
    print("correct index", correct_idx)
    print("wrong index", wrong_idx)
    print("Number of wrong samples:", len(wrong_idx))
    
    # NOTE: X_test is now (100, 1, 64, 64) after resize
    X_test_cpu = X_test.cpu().numpy()
    rd_correct = X_test_cpu[correct_idx, 0]
    rd_wrong = X_test_cpu[wrong_idx, 0]

    # Plot side by side
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(rd_correct[0, :, :], aspect='auto')
    plt.xlabel("Doppler")
    plt.ylabel("Range")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(rd_wrong[0, :, :], aspect='auto')
    plt.xlabel("Doppler")
    plt.ylabel("Range")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # Correctness vs velocities
    correct_idx = np.array(correct_idx, dtype=int)
    wrong_idx = np.array(wrong_idx, dtype=int)
    v_test = np.array(v_test)
    all_labels = np.array(all_labels)

    plt.figure(figsize=(6, 4))
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
    plt.savefig("C:/Users/uig67136/.spyder-py3/python scripts/RadarClassification/Figures/increasingmultitarget/velocity_vs_label_VIT.png", dpi=150)
    plt.show()

    return model