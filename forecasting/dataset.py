from __future__ import annotations
import torch

from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Union


class TimeSeries(Dataset):
    """
    Time series dataset for forecasting anomaly detection algorithm.

    Args:
    X (np.ndarray): Time series data with shape (n_samples, n_variables).
    window_length (int): Length of the window for input data.
    prediction_length (int): Length of the prediction window.
    input_dims (List[int]): List of indices of the variables to be used as input.
    output_dims (List[int]): List of indices of the variables to be used as output.

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: Tuple of input and output data.
    """

    def __init__(self, X, window_length: int, prediction_length: int,
                 input_dims: Optional[List[int]] = None,
                 output_dims: Optional[List[int]] = None,
                 return_index: bool = False
                 ):
        self.input_dims = input_dims or list(range(X.shape[1]))
        self.output_dims = output_dims or list(range(X.shape[1]))
        torch_X = torch.from_numpy(X).float()
        self.index = torch.arange(X.shape[0])
        self.X = torch_X[:, self.input_dims].transpose(0, 1)
        self.Y = torch_X[:, self.output_dims]
        self.n_samples = self.X.shape[1]
        self.window_length = window_length
        self.prediction_length = prediction_length
        self._return_index = return_index

    def get_index(self, return_index: bool) -> TimeSeries:
        if isinstance(return_index, bool):
            self._return_index = return_index
            return self
        else:
            raise ValueError("return_index must be a boolean")

    def __len__(self):
        return self.n_samples - (self.window_length - 1) - self.prediction_length

    def __getitem__(self,
                    index) -> Union[Tuple[torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor],
                                    Tuple[torch.Tensor,
                                          torch.Tensor]]:
        end_idx = index+self.window_length
        x = self.X[:, index:end_idx]
        y = self.Y[end_idx:end_idx+self.prediction_length]
        if self._return_index:
            x_index = self.index[index:end_idx]
            y_index = self.index[end_idx:end_idx+self.prediction_length]
            return x, y, x_index, y_index
        else:
            return x, y
