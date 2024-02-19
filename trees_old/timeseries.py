from __future__ import annotations
import torch
import logging
from typing import List, Optional, Tuple, Union, Callable, Dict
import numpy as np
from ..dataset import BaseTimeSeries
logger = logging.getLogger(__name__)


class TreesTimeSeries(BaseTimeSeries):
    """
    Time series dataset for trees-based anomaly detection algorithm.

    Args:
    X (np.ndarray): Time series data with shape (n_samples, n_variables).
    window_length (int): Length of the window for input data. 
        - Some models only accept unit window_length.
        - For window_legnth more than 1, the model can process sequances instead of point anomalies.
    input_dims (List[int]): List of indices of the variables to be used as input.
    feature_first (bool): If True, the shape of the input data is (n_variables, n_samples).

    Returns:
    torch.Tensor: Input data.
    """

    def __init__(self, *args, **kwargs):
        super(TreesTimeSeries, self).__init__(*args, **kwargs)

    def __len__(self):
        return self.n_samples - (self.window_length - 1)

    def __getitem__(self,
                    index) -> Union[Tuple[torch.Tensor, torch.Tensor,],
                                    torch.Tensor]:
        end_idx = index+self.window_length
        if self.feature_first:
            x = self.X[:, index:end_idx]
        else:
            x = self.X[index:end_idx, :]
        if self._return_index:
            x_index = np.array([index, end_idx, self.step])
            return x, x_index
        else:
            return x
