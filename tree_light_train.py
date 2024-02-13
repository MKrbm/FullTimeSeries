from pytorch_lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from typing import List, cast
import torch
import pandas as pd
import numpy as np
import time
import logging
from sklearn.ensemble import IsolationForest



logger = logging.getLogger(__name__)
window = 100
batch_size = 45
pred_window = 2
df = pd.read_csv('save_csv/synthetic_ts.csv', index_col=0).iloc[:1000]
X = df.values
X = (X - X.mean(axis=0)) / X.std(axis=0)

# ts_data = ForecastingTimeSeries(
#     X,
#     window_length=window,
#     prediction_length=pred_window,
#     feature_first=True)
# train_dl = TimeSeriesDataLoader(ts_data, batch_size=batch_size, shuffle=True)
#
# base_model = DeepAnT(window, pred_window, in_channels=df.shape[1], filter1_size=128,
#                      filter2_size=32, kernel_size=2, pool_size=2, stride=1)
#
# # base_model = LSTMAD(input_size=df.shape[1],
# #                     window=window,
# #                     pred_window=pred_window,
# #                     lstm_layers=3)
#
# model = ForecastingModule(base_model, lr=0.0001)
# #
# trainer = Trainer(
#     max_epochs=2,
#     enable_progress_bar=True,
#     logger=CSVLogger(
#         "logs",
#         name="forecasting/deepant"),
#     log_every_n_steps=10)
# #
# trainer.fit(model, train_dataloaders=train_dl)
#
# test_dl = TimeSeriesDataLoader(ts_data, batch_size=batch_size, shuffle=False)
# with test_dl.get_index():
#     output = trainer.predict(model, test_dl)
#     if isinstance(output, list):
#         anomaly_score_list = [item[1] for item in output]
#         result_index = [item[2] for item in output]
#     else:
#         raise RuntimeError(
#             "The output from the model is expected to be a list of tuple of torch.Tensor")
#
#
# anomaly_score = torch.cat(anomaly_score_list, dim=0).numpy()
# pred_index = torch.cat(result_index, dim=0).numpy()
# anomaly_df = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
# anomaly_df.iloc[pred_index[:, 0]] = anomaly_score
# anomaly_df.to_csv('save_csv/anomaly_score.csv')
