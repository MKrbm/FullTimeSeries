from __future__ import annotations
import torch
import logging
from typing import List, Optional, Tuple, Union, Callable, Dict
from ..dataset import BaseTimeSeries
logger = logging.getLogger(__name__)


class ReconstructionTimeSeries(BaseTimeSeries):
    """
    Time series dataset for reconstruction anomaly detection algorithm.

    Args:
    X (np.ndarray): Time series data with shape (n_samples, n_variables).
    window_length (int): Length of the window for input data.
    input_dims (List[int]): List of indices of the variables to be used as input.
    output_dims (List[int]): List of indices of the variables to be used as output.
    feature_first (bool): If True, the shape of the input data is (n_variables, n_samples).

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: Tuple of input and output data.
    """

    def __init__(self, *args, **kwargs):
        super(ReconstructionTimeSeries, self).__init__(*args, **kwargs)
        if self.feature_first: 
            self.Y = self.Y.transpose(1, 0)

    def __len__(self):
        return self.n_samples - (self.window_length - 1)

    def __getitem__(self,
                    index) -> Union[Tuple[torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor],
                                    Tuple[torch.Tensor,
                                          torch.Tensor]]:
        end_idx = index+self.window_length
        if self.feature_first:
            x = self.X[:, index:end_idx]
            y = self.Y[:, index:end_idx]
        else:
            x = self.X[index:end_idx, :]
            y = self.Y[index:end_idx, :]
        if self._return_index:
            x_index = self.index[index:end_idx]
            y_index = self.index[index:end_idx]
            return x, y, x_index, y_index
        else:
            return x, y
