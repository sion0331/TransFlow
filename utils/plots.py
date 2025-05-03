import matplotlib.pyplot as plt
import numpy as np

def plot_label_distributions(dataset_train, dataset_val, class_labels=(0, 1, 2), title="Label Distribution"):
    y_train = dataset_train.y if isinstance(dataset_train.y, np.ndarray) else dataset_train.y.numpy()
    y_val = dataset_val.y if isinstance(dataset_val.y, np.ndarray) else dataset_val.y.numpy()

    bins = np.arange(min(class_labels) - 0.5, max(class_labels) + 1.5, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Training set
    axes[0].hist(y_train, bins=bins, rwidth=0.8, color='skyblue')
    axes[0].set_title("Training Label Distribution")
    axes[0].set_xlabel("Class Label")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xticks(class_labels)

    # Validation set
    axes[1].hist(y_val, bins=bins, rwidth=0.8, color='salmon')
    axes[1].set_title("Validation Label Distribution")
    axes[1].set_xlabel("Class Label")
    axes[1].set_xticks(class_labels)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_attention(attn_weights, head_idx, title="Attention distribution"):
    """
    attn_weights: Tensor (batch_size, num_heads, seq_len, seq_len)
    head_idx: Which head to plot (0-indexed)
    """
    attn = attn_weights[0, head_idx].detach().cpu().numpy()  # Take first sample in batch
    plt.imshow(attn, cmap="gray")
    plt.title(f"{title} (head {head_idx + 1})")
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.colorbar()
    plt.show()

def plot_training_history(history, title_prefix=""):
    plt.figure(figsize=(10, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix} Loss Over Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{title_prefix} Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()