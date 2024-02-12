import numpy as np
import pandas as pd
import logging
from typing import List, Union, Tuple
from abc import ABC, abstractmethod, abstractproperty

logger = logging.getLogger(__name__)

def sigma(a_score : pd.Series(), sigma : float) -> float:
    if not (a_score > 0).all:
        raise ValueError("Anomal scores must be positive.")

    return a_score.mean() + a_score.std() * sigma

def percentile(a_score : pd.Series(), percent : float = 25):
    x = a_score.values
    if percent > 50 | percent < 0:
        logger.warning("percentile is expected be [0, 50)")

    q1, q3 = np.percentile(x, [percent, 1-percent])
    iqr = q3 - q1
    return q3 + 1.5 * iqr