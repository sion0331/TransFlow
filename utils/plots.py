import matplotlib.pyplot as plt

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