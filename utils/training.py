"""
Description: Training and validation utilities for model training, including performance tracking and checkpoint saving.

This is original code developed for the COMS6998 Deep Learning Final Project.
"""

import torch
import pickle 

def train_validate(model, train_loader, val_loader, optimizer, criterion, epochs, normalization, dataset_type, device):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch + 1}/{epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'./outputs/{dataset_type}/{model.name}_{normalization}.pth')

    f = f'./outputs/{dataset_type}/{model.name}_{normalization}.pkl'
    with open(f, "wb") as f:
        pickle.dump(history, f)
        
    return history


def train(model, loader, optimizer, criterion, device, print_every=500):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (X_batch, y_batch) in enumerate(loader):
        X_batch, y_batch = X_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.int64) 
        optimizer.zero_grad()
        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        c = (predicted == y_batch).sum().item()
        correct += c

        # if i % print_every == 0:
        #     print(f"[Batch {i}/{len(loader)}] Train Loss: {loss.item():.4f} | Train Accuracy: {c/y_batch.size(0):.4f}")
                    
    avg_loss = running_loss / len(loader.dataset)
    avg_accuracy = correct / total

    return avg_loss, avg_accuracy


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.int64) 
            outputs = model(X_batch)
            
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()
    return running_loss / len(loader.dataset), correct / total