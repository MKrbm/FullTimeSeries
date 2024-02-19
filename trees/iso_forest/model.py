from sklearn.ensemble import IsolationForest
import numpy as np
from ..base_model import BaseTreesModel


class ISOF(BaseTreesModel, IsolationForest):
    # Explicitly list all parameters from IsolationForest that ISOF will accept.
    def __init__(self, n_estimators=100, max_samples='auto',  n_jobs=None, random_state=None, warm_start=False):
        # Call the superclass __init__ with all the parameters explicitly.
        IsolationForest.__init__(self, n_estimators=n_estimators, max_samples=max_samples,
                                 n_jobs=n_jobs, random_state=random_state,  warm_start=warm_start)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        # Use IsolationForest's decision_function or score_samples to implement anomaly_score.
        # Here we use score_samples which gives the anomaly score (the higher, the more abnormal).
        # Negate it if you want to maintain consistency with decision_function's semantics
        # where lower values indicate outliers.
        return - self.score_samples(X)
