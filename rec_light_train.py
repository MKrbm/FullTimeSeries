from pytorch_lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from typing import List, cast
import torch
import pandas as pd
import numpy as np
import time
import logging

from seqad.reconstuction.autoencoder import Autoencoder
from seqad.reconstuction import ReconstructionTimeSeries, ReconstructionModule
from seqad.dataset import TimeSeriesDataLoader
logger = logging.getLogger(__name__)
window = 100
batch_size = 45
df = pd.read_csv('save_csv/synthetic_ts.csv', index_col=0).iloc[:1000]
X = df.values
X = (X - X.mean(axis=0)) / X.std(axis=0)
n_features = X.shape[1]

ts_data = ReconstructionTimeSeries(X, window_length=window, feature_first=False)
train_dl = TimeSeriesDataLoader(ts_data, batch_size=batch_size, shuffle=True)
base_model = Autoencoder(input_size=n_features, latent_dim=32)
model = ReconstructionModule(base_model, lr=0.0001)

trainer = Trainer(
    max_epochs=10,
    enable_progress_bar=True,
    logger=CSVLogger(
        "logs",
        name="reconstruction/autoencoder"),
    log_every_n_steps=10)

trainer.fit(model, train_dataloaders=train_dl)

test_dl = TimeSeriesDataLoader(ts_data, batch_size=batch_size, shuffle=False)
with test_dl.get_index():
    output = trainer.predict(model, test_dl)
    if isinstance(output, list):
        anomaly_score_list = [item[1] for item in output]
        result_index = [item[2] for item in output]
    else:
        raise RuntimeError(
            "The output from the model is expected to be a list of tuple of torch.Tensor")

anomaly_score = torch.cat(anomaly_score_list, dim=0).numpy()
pred_index = torch.cat(result_index, dim=0).numpy()
print(pred_index.shape)
anomaly_df = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
anomaly_df.iloc[pred_index[:, 0]] = anomaly_score
anomaly_df.to_csv('save_csv/autoencoder/anomaly_score.csv')
