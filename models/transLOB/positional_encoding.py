import torch
import torch.nn as nn

class LOBPositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, seq_len, channels = x.size()
        position = torch.linspace(-1, 1, steps=seq_len, device=x.device)      # (seq_len,)
        position = position.unsqueeze(0).expand(batch_size, -1).unsqueeze(2)  # (batch_size, seq_len, 1)
        return torch.cat([x, position], dim=2)