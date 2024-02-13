import torch
from torch import nn
import torch.nn.functional as F


class DeepAnT(nn.Module):
    def __init__(
            self,
            window,
            pred_window,
            in_channels,
            filter1_size,
            filter2_size,
            kernel_size,
            pool_size,
            stride):
        super().__init__()
        self.window = window
        self.pred_window = pred_window
        self.in_channels = in_channels
        self.filter1_size = filter1_size
        self.filter2_size = filter2_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.stride = stride

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=filter1_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=0)
        self.conv2 = nn.Conv1d(
            in_channels=filter1_size,
            out_channels=filter2_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=0)
        self.maxpool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(0.25)
        self.dim1 = int(0.5 * (0.5 * (window - 1) - 1)) * filter2_size
        self.lin1 = nn.Linear(self.dim1, in_channels * pred_window)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, self.dim1)
        x = self.dropout(x)
        x = self.lin1(x)
        return x.view(-1, self.pred_window, self.in_channels)
