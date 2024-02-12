from pytorch_lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from typing import List, cast
import torch
import pandas as pd
import numpy as np
import time
import logging

from reconstuction.autoencoder import Autoencoder
from reconstuction.dataset import TimeSeries, TimeSeriesDataLoader, get_predict_index
logger = logging.getLogger(__name__)
window = 100
batch_size = 45
df = pd.read_csv('save_csv/synthetic_ts.csv', index_col=0).iloc[:1000]
X = df.values
X = (X - X.mean(axis=0)) / X.std(axis=0)
n_features = X.shape[1]

ts_data = TimeSeries(X, window_length=window, feature_first=False)
train_dl = TimeSeriesDataLoader(ts_data, batch_size=batch_size, shuffle=True)
model = Autoencoder(input_size=n_features, latent_dim=32, lr=0.005)

trainer = Trainer(
    max_epochs=10,
    enable_progress_bar=True,
    logger=CSVLogger(
        "logs",
        name="reconstruction/autoencoder"),
    log_every_n_steps=10)

trainer.fit(model, train_dataloaders=train_dl)

# n: get anomaly score
test_dl = TimeSeriesDataLoader(ts_data, batch_size=batch_size, shuffle=False)
anomaly_score_list = trainer.predict(model, test_dl)

if not isinstance(anomaly_score_list, list) or \
   not all(isinstance(item, torch.Tensor) for item in anomaly_score_list):
    raise RuntimeError("The anomaly_score_list is expected to be a list of torch.Tensor")
else:
    anomaly_score_list = cast(List[torch.Tensor], anomaly_score_list)
anomaly_score = torch.cat(anomaly_score_list, dim=0)

# n: rescale the anomaly score
# note the all columns are standardized by the same std
# anomaly_score = anomaly_score / anomaly_score.std(axis=0)

# # n: measure the time
pred_index = get_predict_index(test_dl)
anomaly_df = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
anomaly_df.iloc[pred_index] = anomaly_score.numpy()
anomaly_df.to_csv('save_csv/anomaly_score.csv')
