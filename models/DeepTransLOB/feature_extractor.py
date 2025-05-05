import torch
import torch.nn as nn
import torch.nn.functional as F

"""
We apply five one-dimensional convolutional layers to the input X, regarded as a tensor of shape [100,40] (ie. an element of R100 x R40). All layers are dilated causal convolutional layers with 14 features, kernel size 2 and dilation rates 1,2,4,8 and 16 respectively. This means the filter is applied over a window larger than its length by skipping input values with a step given by the dilation rate with each layer respecting the causal order. The first layer with dilation rate 1 corresponds to standard convolution. All activation functions are ReLU.
"""


import torch
import torch.nn as nn


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

        self.conv1 = nn.Sequential(
            CausalConv2d(1, 16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
            CausalConv2d(16, 16, kernel_size=(3, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
            CausalConv2d(16, 16, kernel_size=(3, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16)
        )

        self.conv2 = nn.Sequential(
            CausalConv2d(16, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            CausalConv2d(32, 32, kernel_size=(3, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            CausalConv2d(32, 32, kernel_size=(3, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32)
        )

        self.conv3 = nn.Sequential(
            CausalConv2d(32, 64, kernel_size=(1, 10)),  # collapsing feature dim
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            CausalConv2d(64, 64, kernel_size=(3, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            CausalConv2d(64, 64, kernel_size=(3, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        B, T, F = x.shape  # [B, 100, 40]
        x = x.view(B, 1, T, F)  # [B, 1, 100, 40]
        x = self.conv1(x)       # [B, 16, 100, 20]
        x = self.conv2(x)       # [B, 32, 100, 10]
        x = self.conv3(x)       # [B, 64, 100, 1]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, 100, 1, 64]
        x = x.view(B, T, -1)  # flatten spatial dims: [B, 100, 64]
        return x
        
# class LOBFeatureExtractor2D(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2)),  # [B, 1, 100, 40] -> [B, 32, 100, 20]
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 16, kernel_size=(3, 1), padding=(1, 0)),  # keep time
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 16, kernel_size=(3, 1), padding=(1, 0)),
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(16)
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=(1, 2), stride=(1, 2)),  # [B, 32, 100, 20] -> [B, 32, 100, 10]
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)),
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)),
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(32)
#         )

#         self.conv3 = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=(1, 10)),  # [B, 32, 100, 10] -> [B, 32, 100, 1]
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)),
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)),
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(32)
#         )

#         self.inp1 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 64, kernel_size=(3, 1), padding='same'),
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(64)
#         )

#         self.inp2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 64, kernel_size=(5, 1), padding='same'),
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(64)
#         )

#         self.inp3 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
#             nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(64)
#         )

#     def forward(self, x):
#         B, T, F = x.shape  # (B, 100, 40)
#         x = x.view(B, 1, T, F)  # -> (B, 1, 100, 40)
#         x = self.conv1(x)       # -> (B, 32, 100, 20)
#         # print(x.shape)
#         x = self.conv2(x)       # -> (B, 32, 100, 10)
#         # print(x.shape)
#         x = self.conv3(x)       # -> (B, 32, 100, 1)
#         # print(x.shape)

#         x1 = self.inp1(x)        # (B, 64, 100, 1)
#         x2 = self.inp2(x)        # (B, 64, 100, 1)
#         x3 = self.inp3(x)        # (B, 64, 100, 1)

#         x = torch.cat([x1, x2, x3], dim=1)  # (B, 192, 100, 1)
#         x = x.permute(0, 2, 3, 1).contiguous()  # (B, 100, 1, 192)
#         x = x.view(B, T, -1)  # (B, 100, 192)


        
#         # x = x.permute(0, 2, 3, 1)  # -> (B, 100, 1, 32)
#         # x = x.reshape(B, T, -1)    # -> (B, 100, 32)
#         return x




# class LOBFeatureExtractor2D(nn.Module):
#     def __init__(self, in_channels=1, out_channels=32):
#         super().__init__()
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=(1, 2), stride=(1, 2)),  # reduces feature dim only
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(out_channels),

#             nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),  # same padding on time
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(out_channels),

#             nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),  # same padding again
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm2d(out_channels),
#         )

#     def forward(self, x):
#         # Input: x shape = (B, T, F)
#         B, T, F = x.shape
#         x = x.view(B, 1, T, F)         # -> (B, 1, T, F)
#         x = self.conv_block(x)        # -> (B, C, T, F') because padding preserves T
#         x = x.permute(0, 2, 3, 1)     # -> (B, T, F', C)
#         x = x.reshape(B, T, -1)       # flatten F' and C -> (B, T, F'*C)
#         return x



# class LOBFeatureExtractor(nn.Module):
#     def __init__(self, in_channels, hidden_channels, kernel_size=2):
#         super().__init__()
#         self.kernel_size = kernel_size

#         # Separate price and volume feature branches
#         self.price_conv1 = nn.Conv1d(20, hidden_channels, kernel_size, padding=0, dilation=1)
#         self.price_conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=2)
#         self.price_conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=4)
#         self.price_conv4 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=8)
#         self.price_conv5 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=16)

#         self.volume_conv1 = nn.Conv1d(20, hidden_channels, kernel_size, padding=0, dilation=1)
#         self.volume_conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=2)
#         self.volume_conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=4)
#         self.volume_conv4 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=8)
#         self.volume_conv5 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=16)

#         # Final normalization
#         self.final_norm = nn.LayerNorm(hidden_channels * 2)

#     def causal_pad(self, x, dilation):
#         padding = (self.kernel_size - 1) * dilation
#         return F.pad(x, (padding, 0))

#     def forward_branch(self, x, convs):
#         x = self.causal_pad(x, dilation=1)
#         x = F.relu(convs[0](x))

#         x = self.causal_pad(x, dilation=2)
#         x = F.relu(convs[1](x))

#         x = self.causal_pad(x, dilation=4)
#         x = F.relu(convs[2](x))

#         x = self.causal_pad(x, dilation=8)
#         x = F.relu(convs[3](x))

#         x = self.causal_pad(x, dilation=16)
#         x = F.relu(convs[4](x))

#         return x

#     def forward(self, x):  # (batch, 100, 40)
#         x = x.permute(0, 2, 1)  # (batch, 40, 100)

#         price = x[:, :20, :]   # first 20 channels (price)
#         volume = x[:, 20:, :]  # last 20 channels (volume)

#         price_feat = self.forward_branch(price, [
#             self.price_conv1, self.price_conv2, self.price_conv3, self.price_conv4, self.price_conv5
#         ])
#         volume_feat = self.forward_branch(volume, [
#             self.volume_conv1, self.volume_conv2, self.volume_conv3, self.volume_conv4, self.volume_conv5
#         ])

#         combined = torch.cat([price_feat, volume_feat], dim=1)  # (batch, hidden_channels*2, 100)

#         combined = combined.permute(0, 2, 1)  # (batch, 100, hidden_channels*2)
#         combined = self.final_norm(combined)

#         return combined


        
class LOBFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=2):
        super().__init__()
        self.kernel_size = kernel_size
        
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=0, dilation=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=2)
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=4)
        self.conv4 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=8)
        self.conv5 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=16)
        self.conv6 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=32) # Added
        self.conv7 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=0, dilation=64) # Added

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

        ### Added
        x = self.causal_pad(x, dilation=32)
        x = F.relu(self.conv6(x))

        x = self.causal_pad(x, dilation=64)
        x = F.relu(self.conv7(x))

        x = x.permute(0, 2, 1)
        return x
