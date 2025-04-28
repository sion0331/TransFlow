import torch
import torch.nn as nn
import torch.nn.functional as F

"""
We apply five one-dimensional convolutional layers to the input X, regarded as a tensor of shape [100,40] (ie. an element of R100 x R40). All layers are dilated causal convolutional layers with 14 features, kernel size 2 and dilation rates 1,2,4,8 and 16 respectively. This means the filter is applied over a window larger than its length by skipping input values with a step given by the dilation rate with each layer respecting the causal order. The first layer with dilation rate 1 corresponds to standard convolution. All activation functions are ReLU.
"""

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
