import torch
import torch.nn as nn
from .feature_extractor import LOBFeatureExtractor, LOBFeatureExtractor2D
from .positional_encoding import LOBPositionalEncoding
from .attention_module import LOBTransformerBlock

class DeepTransLOB(nn.Module):

    def __init__(self, num_features=40, num_classes=3, hidden_channels=14, d_model=15, num_heads=3, num_transformer_blocks=2):
        super().__init__()
        self.name = 'DeepTransLOB'

        self.feature_extractor = LOBFeatureExtractor2D() ## Added
        # self.feature_extractor = LOBFeatureExtractor2D(in_channels=1, out_channels=32) ## Added


        self.input_projection = nn.Linear(64, d_model-1) # 640

                # self.feature_extractor = LOBFeatureExtractor2D(num_features, hidden_channels)
        
        # self.layer_norm = nn.LayerNorm(hidden_channels)
        # self.layer_norm = nn.LayerNorm(hidden_channels * 2, hidden_channels*2)
        
        self.position_encoding = LOBPositionalEncoding()
        # self.input_projection = nn.Linear(hidden_channels + 1, d_model)
        # self.input_projection = nn.Linear(hidden_channels *2 + 1, d_model)

        self.transformer_blocks = nn.Sequential(
            LOBTransformerBlock(d_model, num_heads=8),
            LOBTransformerBlock(d_model, num_heads=8),
            LOBTransformerBlock(d_model, num_heads=8),
            LOBTransformerBlock(d_model, num_heads=8)
        )
        # self.transformer_block1 = LOBTransformerBlock(d_model, num_heads)
        # self.transformer_block2 = LOBTransformerBlock(d_model, num_heads)
        # self.transformer_block3 = LOBTransformerBlock(d_model, num_heads)
        # self.transformer_block4 = LOBTransformerBlock(d_model, num_heads)
        
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, return_attention=False): # (b, 100, 60)
        # print('input: ', x.shape)
        x = self.feature_extractor(x)             # (b, 100, 14)
        # print('feature_extractor: ', x.shape)
        
        # x = self.layer_norm(x)                    # (b, 100, 14)
        # print('layer_norm: ', x.shape)
        x = self.input_projection(x)
        # print('input_projection: ', x.shape)
        
        x = self.position_encoding(x)             # (b, 100, 15)
        # print('position_encoding: ', x.shape)

        x = self.transformer_blocks(x) 
        
        # x = self.input_projection(x)              # (b, 100, 64)
        
        # x = self.transformer_block1(x)             # (b, 100, 15)
        # x = self.transformer_block2(x)             # (b, 100, 15)
        # x = self.transformer_block3(x)             # (b, 100, 15)
        # x = self.transformer_block4(x)             # (b, 100, 15)
        x = x[:, -1, :]                           # (b, 64)
        x = self.fc_out(x)                        # (b, 3)
        return x