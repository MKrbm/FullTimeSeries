from sklearn.base import clone
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import numpy as np
from .timeseries import TreesTimeSeries
from .base_model import BaseTreesModel
from ..dataset import TimeSeriesDataLoader
import logging
logger = logging.getLogger(__name__)


class TreesTrainer(ABC):
    model: BaseTreesModel
    trained: bool

    def __init__(self, model: BaseTreesModel):
        if not isinstance(model, BaseTreesModel):
            raise ValueError("Model is not instance of BaseTreesModel")
        self.model = clone(model)
        self.trained = False

    def predict(self, ts: TreesTimeSeries, batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return anomaly_score and index"""
        if not self.trained:
            raise ValueError("Please train the model first")
        n_samples = len(ts)
        window = ts.window_length
        if batch_size is None:
            batch_size = int(n_samples / window)
        test_dl = TimeSeriesDataLoader(
            ts, batch_size=batch_size, shuffle=False)
        score_list = []
        index_list = []
        with test_dl.get_index():
            for X, index in test_dl:
                X = X.numpy()
                X = X.reshape(X.shape[0], -1)
                score_list.append(self.model.anomaly_score(X))
                index_list.append(index[:, 0].numpy())
        return np.concatenate(score_list), np.concatenate(index_list)

    def fit(self, ts: TreesTimeSeries) -> BaseTreesModel:
        if self.is_trained():
            return

        n_samples = len(ts)
        try:
            train_dl = TimeSeriesDataLoader(
                ts, batch_size=n_samples, shuffle=True)
            X = next(iter(train_dl))
            X = X.numpy().reshape(n_samples, -1)
            self.model.fit(X)
        except MemoryError:
            logger.exception("Insufficient memory for full-dataset fitting.")
        except Exception:
            logger.exception("Unknown error occured")
        self.trained = True
        return self.model

    def fit_batch(self, ts: TreesTimeSeries, batch_size: Optional[int] = None, epochs=1) -> BaseTreesModel:
        if self.is_trained():
            return

        self.assert_warm_start()
        n_samples = len(ts)
        n_estimators = self.model.n_estimators
        if batch_size is None:
            batch_size = int(n_samples / n_estimators) * epochs
            self.model.n_estimators = 1
        else:
            if self.model.n_estimators != 1:
                logger.warning(
                    "n_estimators in the model is expected to be 1 before fitting with batch learning.")
        train_dl = TimeSeriesDataLoader(
            ts, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for X in train_dl:
                X = X.numpy()
                if (X.shape[0] < self.model.max_samples):
                    continue
                X = X.reshape(X.shape[0], -1)
                self.model.fit(X)
                self.model.n_estimators += 1

        self.trained = True
        return self.model

    def reset(self) -> bool:
        self.model = clone(self.model)
        self.trained = False

    def is_trained(self) -> bool:
        if self.trained:
            logger.info("Train is already done."
                        "Please reset the model before running the fitting again.")
            return True
        else:
            return False

    def assert_warm_start(self) -> bool:
        """
        Check if the given sklearn ensemble model has a 'warm_start' attribute
        and if it is set to True.

        Parameters:
        - model: The model to check, expected to be an instance of a scikit-learn estimator.

        Returns:
        - True if 'warm_start' exists and is True, False otherwise.
        """
        # Check if 'warm_start' attribute exists
        if hasattr(self.model, 'warm_start'):
            # Check if 'warm_start' is set to True
            if not self.model.warm_start:
                raise ValueError(
                    "The model has warm_start but it's not activated")
            if not hasattr(self.model, 'n_estimators'):
                raise ValueError(
                    "Model is expected to have n_estimatros attributes")
        else:
            # 'warm_start' does not exist
            raise ValueError("The model doesn't have warm_start attributes."
                             "Batch Learning is not available for the model")
