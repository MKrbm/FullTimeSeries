from sklearn.ensemble import BaseEnsemble
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import logging
import numpy as np
logger = logging.getLogger(__name__)


class BaseTreesModel(BaseEnsemble, ABC):

    @abstractmethod
    def anomaly_score(X: np.ndarray) -> np.ndarray:
        """
        Anomaly score should indicate anomalous point if the score is higher.
        """
        pass
