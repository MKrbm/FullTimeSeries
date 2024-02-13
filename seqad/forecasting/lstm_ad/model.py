import torch
from torch import nn


class LSTMAD(nn.Module):
    def __init__(self, input_size: int, lstm_layers: int, window: int, pred_window: int):
        super().__init__()
        self.input_size = input_size
        self.lstm_layers = lstm_layers
        self.window = window
        self.pred_window = pred_window
        self.hidden_units = input_size

        self.lstms = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_units * self.pred_window,
            batch_first=True,
            num_layers=lstm_layers)
        self.dense = nn.Linear(
            in_features=self.window * self.hidden_units * self.pred_window,
            out_features=self.hidden_units * self.pred_window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, hidden = self.lstms(x)
        x = x.reshape(-1, self.window * self.hidden_units * self.pred_window)
        x = self.dense(x).reshape(-1, self.pred_window, self.hidden_units)
        return x


