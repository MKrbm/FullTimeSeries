from pytorch_lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
import torch
import pandas as pd
import logging
from typing import List, cast

from forecasting.DeepAnT import DeepAnT
from forecasting.dataset import TimeSeries, TimeSeriesDataLoader
from forecasting.anomaly_score import anomaly_score_single

logger = CSVLogger("logs", name="deepant")
window = 100
batch_size = 45
pred_window = 2
df = pd.read_csv('save_csv/synthetic_ts.csv', index_col=0).iloc[:1000]
X = df.values
# normalizing the data
X = (X - X.mean(axis=0)) / X.std(axis=0)
# print(X.mean(axis=0))

ts_data = TimeSeries(X, window_length=window, prediction_length=pred_window)
train_dl = TimeSeriesDataLoader(ts_data, batch_size=batch_size, shuffle=True)
model = DeepAnT(window, pred_window, in_channels=df.shape[1], filter1_size=128,
                filter2_size=32, kernel_size=2, pool_size=2, stride=1)

trainer = Trainer(max_epochs=2, enable_progress_bar=True, logger=logger, log_every_n_steps=10)

trainer.fit(model, train_dataloaders=train_dl)

test_dl = TimeSeriesDataLoader(ts_data, batch_size=batch_size, shuffle=False)
anomaly_score_list = trainer.predict(model, test_dl)

# Ensure all elements are Tensors (runtime check)
if not isinstance(anomaly_score_list, list) or \
   not all(isinstance(item, torch.Tensor) for item in anomaly_score_list):
    raise RuntimeError("The anomaly_score_list is expected to be a list of torch.Tensor")
else:
    anomaly_score_list = cast(List[torch.Tensor], anomaly_score_list)
anomaly_score = torch.stack(anomaly_score_list, dim=0)

result_index = []
with test_dl.get_index():
    for i, (X, y, x_i, y_i) in enumerate(test_dl):
        result_index.append(y_i)
pred_index = torch.cat(result_index, dim=0)
