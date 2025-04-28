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


        
#### With dropouts
# class LOBTransformerBlock(nn.Module):
#     def __init__(self, d_model, num_heads, dropout=0.1):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)

#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model * 4),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * 4, d_model),
#         )
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, x):
#         attn_output, _ = self.attention(x, x, x, attn_mask=self._generate_causal_mask(x))
#         x = x + self.dropout1(attn_output)
#         x = self.norm1(x)

#         ffn_output = self.ffn(x)
#         x = x + self.dropout2(ffn_output)
#         x = self.norm2(x)

#         return x
    
#     def _generate_causal_mask(self, x):
#         seq_len = x.size(1)
#         mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
#         return mask


        
# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, d_model, num_heads, masking=True):
#         super().__init__()
#         assert d_model % num_heads == 0

#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
#         self.masking = masking

#         self.q_linear = nn.Linear(d_model, d_model)
#         self.k_linear = nn.Linear(d_model, d_model)
#         self.v_linear = nn.Linear(d_model, d_model)
#         self.out_linear = nn.Linear(d_model, d_model)

#     def forward(self, x):
#         batch_size, seq_len, d_model = x.size()

#         Q = self.q_linear(x)
#         K = self.k_linear(x)
#         V = self.v_linear(x)

#         Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq, d_k)
#         K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

#         scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (batch, heads, seq, seq)

#         if self.masking:
#             mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(x.device)
#             scores = scores.masked_fill(mask == 1, float('-inf'))

#         attn = F.softmax(scores, dim=-1)
#         out = torch.matmul(attn, V)

#         out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
#         out = self.out_linear(out)
#         return out