import torch
from .dataset import TimeSeries
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
import numpy as np


def anomaly_score_single(y_pred: torch.Tensor,
                         ts_data: TimeSeries) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = y_pred.shape[0]
    _, y_test, _, y_test_index = next(
        iter(DataLoader(ts_data.get_index(True), batch_size=n_samples, shuffle=False)))

    # calculate euclidian distance
    anomaly_score = torch.sqrt(
        torch.nn.functional.mse_loss(
            y_pred.detach(),
            y_test.detach(),
            reduction="none").sum(
            dim=[
                1,
                2]))
    # standardize error
    anomaly_score = (anomaly_score - anomaly_score.mean()).abs() / anomaly_score.std()
    return anomaly_score.numpy().reshape(n_samples, 1), y_test_index.numpy()
