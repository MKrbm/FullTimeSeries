from __future__ import annotations
import torch
from contextlib import contextmanager

import logging
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Union, Callable, Dict
logger = logging.getLogger(__name__)


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
                 feature_first: bool = False,
                 step: int = 1,
                 ):
        self.input_dims = input_dims or list(range(X.shape[1]))
        self.output_dims = output_dims or list(range(X.shape[1]))
        self.feature_first = feature_first
        torch_X = torch.from_numpy(X).float()
        self.X = torch_X[:, self.input_dims]
        self.n_samples = self.X.shape[0]
        self.index = torch.arange(X.shape[0])
        self.Y = torch_X[:, self.output_dims]
        self.window_length = window_length
        self.prediction_length = prediction_length
        self._return_index = False
        self.step = step
        if feature_first:
            self.X = self.X.transpose(1, 0)

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
        if self.feature_first:
            x = self.X[:, index:end_idx:self.step]
        else:
            x = self.X[index:end_idx:self.step, :]
        y = self.Y[end_idx:pred_idx:self.step]
        if self._return_index:
            x_index = self.index[index:end_idx:self.step]
            y_index = self.index[end_idx:pred_idx:self.step]
            return x, y, x_index, y_index
        else:
            return x, y


class TimeSeriesDataLoader(DataLoader):
    def __init__(self, dataset: TimeSeries, *args, **kwargs):
        super(TimeSeriesDataLoader, self).__init__(dataset, *args, **kwargs)
        if not isinstance(dataset, TimeSeries):
            raise ValueError("First argument must be TimeSeries")

    @contextmanager
    def get_index(self):
        """
        Context manager to enable index returning in the TimeSeries dataset.
        """
        self.dataset._return_index = True
        try:
            yield
        finally:
            self.dataset._return_index = False


def get_predict_index(tsdl: TimeSeriesDataLoader) -> np.ndarray:

    start = time.time()
    result_index = []
    with tsdl.get_index():
        for i, (X, y, x_i, y_i) in enumerate(tsdl):
            result_index.append(y_i)
    pred_index = torch.cat(result_index, dim=0)
    if (pred_index.shape[1] != 1):
        logger.warning("The shape output from the model is expected to be [1, # of columns]"
                       "Automatically selecting the first column as the prediction index.")
    end = time.time()
    logger.info(f"Time to get prediction index : {end - start}")
    logger.debug(f"Prediction index shape: {pred_index.shape}")
    return pred_index[:, 0].numpy()
