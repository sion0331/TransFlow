"""
Author: Sion Chun
Description: Visualization utilities for inspecting LOB dataset and model performance.
- plot_label_distributions: Show distribution of class labels in training and validation datasets.
- plot_training_history: Plot loss and accuracy curves over training epochs.

This is original code written for the COMS6998 Deep Learning Final Project.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_label_distributions(dataset_train, dataset_val, class_labels=(0, 1, 2), title="Label Distribution"):
    def extract_labels(subset):
        if hasattr(subset, 'dataset') and hasattr(subset, 'indices'):
            return np.array([subset.dataset[i][1].item() for i in subset.indices])
        elif hasattr(subset, 'y'):
            return subset.y.numpy() if hasattr(subset.y, 'numpy') else subset.y
        else:
            raise ValueError("Unsupported dataset type")

    y_train = extract_labels(dataset_train)
    y_val   = extract_labels(dataset_val)

    bins = np.arange(min(class_labels) - 0.5, max(class_labels) + 1.5, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    axes[0].hist(y_train, bins=bins, rwidth=0.8, color='skyblue')
    axes[0].set_title("Training Label Distribution")
    axes[0].set_xlabel("Class Label")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xticks(class_labels)

    axes[1].hist(y_val, bins=bins, rwidth=0.8, color='salmon')
    axes[1].set_title("Validation Label Distribution")
    axes[1].set_xlabel("Class Label")
    axes[1].set_xticks(class_labels)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_training_history(history, title_prefix=""):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix} Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{title_prefix} Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()