from __future__ import annotations
import torch
from contextlib import contextmanager

import logging
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Union, Callable, Dict
from ..dataset import BaseTimeSeries 
logger = logging.getLogger(__name__)


class ForecastingTimeSeries(BaseTimeSeries):
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
    X: (window_length, n_features) or (n_features, window_length) if feature_first is True
    Y: (prediction_length, n_features)
    """

    def __init__(self, *args, prediction_length: int = 1, **kwargs):
        super(ForecastingTimeSeries, self).__init__(*args, **kwargs)
        self.prediction_length = prediction_length

    def __len__(self):
        return self.n_samples - (self.window_length*self.step - 1) - self.prediction_length

    def __getitem__(self,
                    index) -> Union[Tuple[torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor],
                                    Tuple[torch.Tensor,
                                          torch.Tensor]]:
        end_idx = index+self.window_length*self.step
        pred_idx = end_idx+self.prediction_length*self.step
        y = self.Y[end_idx:pred_idx:self.step]
        if self.feature_first:
            x = self.X[:, index:end_idx:self.step]
        else:
            x = self.X[index:end_idx:self.step, :]
        if self._return_index:
            x_index = self.index[index:end_idx:self.step]
            y_index = self.index[end_idx:pred_idx:self.step]
            return x, y, x_index, y_index
        else:
            return x, y
