import torch
import torch.nn as nn

class LOBPositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, channels, seq_len = x.size()
        position = torch.linspace(0, 1, steps=seq_len, device=x.device)
        position = position.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
        return torch.cat([x, position], dim=1)
