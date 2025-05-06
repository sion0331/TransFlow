"""
We apply five one-dimensional convolutional layers to the input X, regarded as a tensor of shape [100,40] (ie. an element of R100 x R40). All layers are dilated causal convolutional layers with 14 features, kernel size 2 and dilation rates 1,2,4,8 and 16 respectively. This means the filter is applied over a window larger than its length by skipping input values with a step given by the dilation rate with each layer respecting the causal order. The first layer with dilation rate 1 corresponds to standard convolution. All activation functions are ReLU.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class LOBFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=2):
        super().__init__()
        self.kernel_size = kernel_size
        
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=0, dilation=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=2)
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=4)
        self.conv4 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=8)
        self.conv5 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=16)

    def causal_pad(self, x, dilation):
        padding = (self.kernel_size - 1) * dilation
        return F.pad(x, (padding, 0))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = self.causal_pad(x, dilation=1)
        x = F.relu(self.conv1(x))

        x = self.causal_pad(x, dilation=2)
        x = F.relu(self.conv2(x))

        x = self.causal_pad(x, dilation=4)
        x = F.relu(self.conv3(x))

        x = self.causal_pad(x, dilation=8)
        x = F.relu(self.conv4(x))

        x = self.causal_pad(x, dilation=16)
        x = F.relu(self.conv5(x))

        x = x.permute(0, 2, 1)
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
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, return_attention=False):
        attn_output, attn_weights = self.attention(x, x, x, attn_mask=self._generate_causal_mask(x))
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

        
class TransLOB(nn.Module):

    def __init__(self, num_features=40, num_classes=3, hidden_channels=14, d_model=15, num_heads=3, num_transformer_blocks=2):
        super().__init__()
        self.name = 'TransLOB'
        
        self.feature_extractor = LOBFeatureExtractor(num_features, hidden_channels)
        self.layer_norm = nn.LayerNorm(hidden_channels)
        self.position_encoding = LOBPositionalEncoding()
        self.transformer_block1 = LOBTransformerBlock(d_model, num_heads)
        self.transformer_block2 = LOBTransformerBlock(d_model, num_heads)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, return_attention=False): # (b, 100, 60)
        x = self.feature_extractor(x)             # (b, 100, 14)
        x = self.layer_norm(x)                    # (b, 100, 14)
        x = self.position_encoding(x)             # (b, 100, 15)
        x = self.transformer_block1(x)             # (b, 100, 15)
        x = self.transformer_block2(x)             # (b, 100, 15)
        x = x[:, -1, :]                           # (b, 64)
        x = self.fc_out(x)                        # (b, 3)
        return x
        