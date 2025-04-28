import torch
import torch.nn as nn
from .feature_extractor import LOBFeatureExtractor
from .positional_encoding import LOBPositionalEncoding
from .attention_module import LOBTransformerBlock

class TransLOB(nn.Module):

    def __init__(self, num_features, num_classes, hidden_channels=14, d_model=15, num_heads=3, num_transformer_blocks=2):
        super().__init__()
        self.feature_extractor = LOBFeatureExtractor(num_features, hidden_channels)
        self.layer_norm = nn.LayerNorm(hidden_channels)
        self.position_encoding = LOBPositionalEncoding()
        self.input_projection = nn.Linear(hidden_channels + 1, d_model)
        self.transformer_block = LOBTransformerBlock(d_model, num_heads=3)
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
        x = self.input_projection(x)              # (b, 100, 64)
        x = self.transformer_block(x)             # (b, 100, 15)
        x = self.transformer_block(x)             # (b, 100, 15)
        x = x[:, -1, :]                           # (b, 64)
        x = self.fc_out(x)                        # (b, 3)
        return x
        