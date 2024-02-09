import torch
from torch.utils.data import DataLoader
import pandas as pd
from typing import List
from torch.utils.data import Dataset
import numpy as np
import logging

from forecasting.DeepAnT import DeepAnT
from forecasting.dataset import TimeSeries
from forecasting.anomaly_score import anomaly_score_single

logger = logging.getLogger(__name__)

window = 100
batch_size = 45
pred_window = 2
df = pd.read_csv('synthetic_ts.csv', index_col=0)
X = df.values[:1000]
# normalizing the data
X = (X - X.mean(axis=0)) / X.std(axis=0)
# print(X.mean(axis=0))

ts_data = TimeSeries(X, window_length=window, prediction_length=pred_window)
dataloader_train = DataLoader(ts_data, batch_size=batch_size, shuffle=False)
model = DeepAnT(window, pred_window, in_channels=df.shape[1], filter1_size=128,
                filter2_size=32, kernel_size=2, pool_size=2, stride=1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# criterion = torch.nn.L1Loss()
criterion = torch.nn.MSELoss()

for epoch in range(1):
    model.train()
    train_losses: List[float] = []
    for i, (X, y) in enumerate(dataloader_train):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    print(sum(train_losses))

result = []
result_index = []
model.eval()
for X, y, x_i, y_i in DataLoader(ts_data.get_index(True), batch_size=batch_size, shuffle=False):
    out = model(X).detach()
    if out.shape[1] != 1:
        # raise RuntimeError("The shape output from the model is expected to be [1, # of columns]")
        logger.error("The shape output from the model is expected to be [1, # of columns]")
    result.append(out)
    result_index.append(y_i)

y_pred_index = torch.cat(result_index, dim=0)
y_pred = torch.cat(result, dim=0)

ano_score, ano_index = anomaly_score_single(y_pred, ts_data)
print(ano_score.shape)
if ano_index.shape[0] != y_pred_index.shape[0]:
    raise RuntimeError("The shape of the index and the anomaly score should be the same")
if (ano_index != y_pred_index.numpy()).any():
    raise RuntimeError("The index and the anomaly score should be the same")

if ano_index.shape[1] != 1:
    logger.warning("The anomaly index is expected to be of shape [n, 1]"
                   " Only the first column will be used for the anomaly score.")
    ano_index = ano_index[:, 0]


anomaly_df = pd.DataFrame(np.nan, index=df.index, columns=["anomaly_score"], dtype=float)
if anomaly_df.iloc[ano_index].shape != ano_score.shape:
    logger.error(
        f"The shape of the anomaly score is {anomaly_df.iloc[ano_index].shape}"
        f" and the shape of the index is {ano_score.shape}")
anomaly_df.iloc[ano_index] = ano_score
anomaly_df.to_csv('anomaly_score.csv')
