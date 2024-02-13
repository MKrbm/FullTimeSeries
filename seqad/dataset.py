from __future__ import annotations
import torch
from contextlib import contextmanager

import logging
import time
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Union, Callable, cast
from abc import ABC, abstractmethod
logger = logging.getLogger(__name__)


class BaseTimeSeries(Dataset, ABC):
    """
    Time series dataset for forecasting/reconstruction anomaly detection algorithm.

    Args:
    X (np.ndarray): Time series data with shape (n_samples, n_variables).
    window_length (int): Length of the window for input data.
    prediction_length (int): Length of the prediction window.
    input_dims (List[int]): List of indices of the variables to be used as input.
    output_dims (List[int]): List of indices of the variables to be used as output.
    feature_first (bool): If True, the shape of the input data is (n_variables, n_samples).
    step (int): Step size for the input data.

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: Tuple of input and output data.
    """

    def __init__(self, X, window_length: int,
                 input_dims: Optional[List[int]] = None,
                 output_dims: Optional[List[int]] = None,
                 feature_first: bool = False,
                 step: int = 1,
                 ):
        self.window_length = window_length
        self._return_index = False
        self.step = step
        self.input_dims = input_dims or list(range(X.shape[1]))
        self.output_dims = output_dims or list(range(X.shape[1]))
        self.feature_first = feature_first
        torch_X = torch.from_numpy(X).float()
        self.X = torch_X[:, self.input_dims]
        self.Y = torch_X[:, self.output_dims]
        self.n_samples = self.X.shape[0]
        self.index = torch.arange(X.shape[0])
        if feature_first:
            self.X = self.X.transpose(1, 0)

    @abstractmethod
    def __len__(self):
        """Return the length of the dataset.

        For the reconstruction dataset, the length is
        n_samples - (window_length - 1).
        For the forecasting dataset, the length is
        n_samples - (window_length*step - 1) - prediction_length.
        """
        pass

    @abstractmethod
    def __getitem__(self,
                    index) -> Union[Tuple[torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor],
                                    Tuple[torch.Tensor,
                                          torch.Tensor]]:
        """Get the input and output data for the given index.

        If the feature_first is True,
        the shape of the input data is (n_variables, n_samples).
        If _return_index is True, the index of the output data is also returned.
        """
        pass


class TimeSeriesDataLoader(DataLoader):
    def __init__(self, dataset: BaseTimeSeries, *args, **kwargs):
        self.dataset = dataset
        super(TimeSeriesDataLoader, self).__init__(dataset, *args, **kwargs)
        if not isinstance(dataset, BaseTimeSeries):
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


def get_anomaly_df(df: pd.DataFrame, score: torch.Tensor, timeseries: BaseTimeSeries) -> pd.DataFrame:
    start = time.time()
    result_index = []
    tsdl = TimeSeriesDataLoader(timeseries, batch_size=score.shape[0], shuffle=False)
    with tsdl.get_index():
        x, y, x_index, pred_index = next(iter(tsdl))
    if (pred_index.shape[1] != 1):
        logger.warning("The shape output from the model is expected to be [1, # of columns]"
                       "Automatically selecting the first column as the prediction index.")

    anomaly_df = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    anomaly_df.iloc[pred_index[:, 0].numpy()] = score.numpy()

    end = time.time()
    logger.info(f"Time to get prediction index : {end - start}")
    logger.debug(f"Prediction index shape: {pred_index.shape}")
    return anomaly_df
