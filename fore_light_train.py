from pytorch_lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from typing import List, cast
import torch
import pandas as pd
import numpy as np
import time
import logging

from forecasting.deepant import DeepAnT
from forecasting.lstm_ad import LSTMAD
# from forecasting.dataset import TimeSeries, TimeSeriesDataLoader, get_predict_index
from reconstuction.autoencoder import Autoencoder
from reconstuction.dataset import TimeSeries, TimeSeriesDataLoader, get_predict_index
logger = logging.getLogger(__name__)
window = 100
batch_size = 45
pred_window = 2
df = pd.read_csv('save_csv/synthetic_ts.csv', index_col=0).iloc[:1000]
X = df.values
# normalizing the data
X = (X - X.mean(axis=0)) / X.std(axis=0)
# print(X.mean(axis=0))

ts_data = TimeSeries(X, window_length=window, prediction_length=pred_window, feature_first=False)
train_dl = TimeSeriesDataLoader(ts_data, batch_size=batch_size, shuffle=True)
# model = DeepAnT(window, pred_window, in_channels=df.shape[1], filter1_size=128,
#                 filter2_size=32, kernel_size=2, pool_size=2, stride=1)
# model = LSTMAD(
#     input_size=df.shape[1],
#     lstm_layers=1,
#     window=window,
#     pred_window=pred_window,
#     lr=0.0001)

model = Autoencoder(input_size=df.shape[1], latent_dim=32, learning_rate=0.005)

trainer = Trainer(
    max_epochs=2,
    enable_progress_bar=True,
    logger=CSVLogger(
        "logs",
        name="deepant"),
    log_every_n_steps=10)

trainer.fit(model, train_dataloaders=train_dl)

test_dl = TimeSeriesDataLoader(ts_data, batch_size=batch_size, shuffle=False)
anomaly_score_list = trainer.predict(model, test_dl)

if not isinstance(anomaly_score_list, list) or \
   not all(isinstance(item, torch.Tensor) for item in anomaly_score_list):
    raise RuntimeError("The anomaly_score_list is expected to be a list of torch.Tensor")
else:
    anomaly_score_list = cast(List[torch.Tensor], anomaly_score_list)
anomaly_score = torch.cat(anomaly_score_list, dim=0)
#
# # n: measure the time
pred_index = get_predict_index(test_dl)
print(pred_index)
anomaly_df = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
anomaly_df.iloc[pred_index] = anomaly_score.numpy()
print(anomaly_df.tail())
