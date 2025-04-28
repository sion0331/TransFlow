import torch
import torch.nn as nn

import torch
import torch.nn as nn

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