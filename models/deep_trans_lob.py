"""
Custom PyTorch implementation of DeepTransLOB – a hybrid model combining 2D Causal Convolutions and Transformer layers (based on https://github.com/jwallbridge/translob)

Key modifications:
- Rewritten in PyTorch from the original TensorFlow version
- CausalConv2d: Manually applies causal padding along the time axis for 2D convolutions
- LOBFeatureExtractor2D: Applies dilated causal convolutions to extract spatial-temporal features
- LOBTransformerBlock: Transformer block with causal attention masking and dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=(1, 1), stride=(1, 1)):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=(0, 0), dilation=dilation  # padding manually handled
        )

    def forward(self, x):
        # Causal pad only on the left (time axis = 2)
        pad_time = (self.kernel_size[0] - 1) * self.dilation[0]
        x = F.pad(x, (0, 0, pad_time, 0))  # (pad_W_left, pad_W_right, pad_H_top, pad_H_bottom)
        return self.conv(x)


class LOBFeatureExtractor2D(nn.Module):
    def __init__(self):
        super().__init__()

        dim = 64
        self.conv1 = nn.Sequential(
            CausalConv2d(1, dim, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            CausalConv2d(dim, dim, kernel_size=(3, 1), dilation=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(dim),
            CausalConv2d(dim, dim, kernel_size=(3, 1), dilation=(2, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(dim)
        )

        self.conv2 = nn.Sequential(
            CausalConv2d(dim, dim, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(dim),
            CausalConv2d(dim, dim, kernel_size=(3, 1), dilation=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(dim),
            CausalConv2d(dim, dim, kernel_size=(3, 1), dilation=(8, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(dim)
        )
        
        self.conv3 = nn.Sequential(
            CausalConv2d(dim, dim, kernel_size=(1, 10), stride=(1, 10)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        B, T, F = x.shape                       # [B, 100, 40]
        x = x.view(B, 1, T, F)                  # [B, 1, 100, 40]
        x = self.conv1(x)                       # [B, 16, 100, 20]
        x = self.conv2(x)                       # [B, 32, 100, 10]
        x = self.conv3(x)                       # [B, 64, 100, 1]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, 100, 1, 64]
        x = x.view(B, T, -1)                    # [B, 100, 64]
        return x
        

class LOBPositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, seq_len, channels = x.size()
        position = torch.linspace(-1, 1, steps=seq_len, device=x.device)      # (seq_len,)
        position = position.unsqueeze(0).expand(batch_size, -1).unsqueeze(2)  # (batch_size, seq_len, 1)
        return torch.cat([x, position], dim=2)


class LOBTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, return_attention=False):
        attn_output, attn_weights = self.attention(x, x, x, attn_mask=self._generate_causal_mask(x))
        attn_output = self.dropout1(attn_output) 
        
        x = x + attn_output
        x = self.norm1(x)
    
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
    
        if return_attention:
            return x, attn_weights
        else:
            return x
            
    def _generate_causal_mask(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        return mask
    
        
class DeepTransLOB(nn.Module):

    def __init__(self, num_features=40, num_classes=3, hidden_channels=64, d_model=64, num_heads=4, num_transformer_blocks=4):
        super().__init__()
        self.name = 'DeepTransLOB'

        self.feature_extractor = LOBFeatureExtractor2D()
        self.input_projection = nn.Linear(hidden_channels, d_model-1)
        self.position_encoding = LOBPositionalEncoding()
        self.transformer_blocks = nn.Sequential(
            LOBTransformerBlock(d_model, num_heads),
            LOBTransformerBlock(d_model, num_heads),
            LOBTransformerBlock(d_model, num_heads),
            LOBTransformerBlock(d_model, num_heads)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, return_attention=False): # (b, 100, 40)
        x = self.feature_extractor(x)             # (b, 100, 64)
        x = self.input_projection(x)              # (b, 100, 63)
        x = self.position_encoding(x)             # (b, 100, 64)
        x = self.transformer_blocks(x)            # (b, 100, 64)
        x = x[:, -1, :]                           # (b, 64)
        x = self.fc_out(x)                        # (b, 3)
        return x