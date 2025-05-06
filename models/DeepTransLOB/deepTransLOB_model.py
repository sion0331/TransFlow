import torch
import torch.nn as nn
from .feature_extractor import LOBFeatureExtractor, LOBFeatureExtractor2D
from .positional_encoding import LOBPositionalEncoding
from .attention_module import LOBTransformerBlock

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