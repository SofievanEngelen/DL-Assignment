import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class FCLayer(nn.Module):
    def __init__(self, input_size, output_size, batch_norm=True, dropout=False):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size) if batch_norm else None
        self.dropout = nn.Dropout(0.5) if dropout else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, padding=1, stride=2, batch_norm=True):
        super(ConvLayer, self).__init__()
        self.batch_norm = batch_norm
        self.conv = nn.Conv1d(input_size, output_size, kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm1d(output_size) if batch_norm else None
        self.pool = nn.MaxPool1d(kernel_size, stride)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        self.convlayer1 = ConvLayer(1, 16)
        self.convlayer2 = ConvLayer(16, 64)

        # Calculate the size of the flattened layer after convolutions
        conv_output_size = self._get_conv_output_size(input_size)

        self.fclayer3 = FCLayer(conv_output_size, 512)
        self.fclayer4 = FCLayer(512, 256)
        self.fclayer5 = FCLayer(256, 128)
        self.fclayer6 = FCLayer(128, 64)
        self.fclayer7 = FCLayer(64, 32)
        self.fclayer8 = FCLayer(32, 1, batch_norm=False, dropout=True)

    def _get_conv_output_size(self, input_size):
        # Create a dummy tensor with the same shape as the input data
        x = torch.zeros(1, 1, input_size)
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for convolution
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = x.view(x.size(0), -1)  # Flatten the output for fully connected layers

        x = self.fclayer3(x)
        x = self.fclayer4(x)
        x = self.fclayer5(x)
        x = self.fclayer6(x)
        x = self.fclayer7(x)
        x = self.fclayer8(x)

        return x
