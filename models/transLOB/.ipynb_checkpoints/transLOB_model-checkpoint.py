import torch
import torch.nn as nn
from .feature_extractor import LOBFeatureExtractor
from .positional_encoding import LOBPositionalEncoding
from .attention_module import LOBTransformerBlock

class TransLOB(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=14, d_model=64, num_heads=4, num_transformer_blocks=2):
        super().__init__()
        self.feature_extractor = LOBFeatureExtractor(num_features, hidden_channels)
        self.position_encoding = LOBPositionalEncoding()
        self.input_projection = nn.Linear(hidden_channels + 1, d_model)

        self.transformer_blocks = nn.Sequential(
            *[LOBTransformerBlock(d_model, num_heads) for _ in range(num_transformer_blocks)]
        )

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        x = self.position_encoding(x)
        x = x.permute(0, 2, 1)
        x = self.input_projection(x)
        x = self.transformer_blocks(x)
        x = x[:, -1, :]
        x = self.fc_out(x)
        return x
