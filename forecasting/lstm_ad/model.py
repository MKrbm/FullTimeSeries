import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np
from torch.optim import Optimizer as BaseOptimizer
from torch.optim import Adam, SGD, RMSprop
from torch.utils.data import DataLoader
from typing import List, Tuple
from enum import Enum
import os
import logging

logger = logging.getLogger(__name__)


class LSTMAD(pl.LightningModule):
    def __init__(self,
                 input_size: int,
                 lstm_layers: int,
                 window: int,
                 pred_window: int,
                 lr: float,
                 *args, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.lstm_layers = lstm_layers
        self.window = window
        self.pred_window = pred_window
        self.hidden_units = input_size
        self.criterion = nn.functional.mse_loss
        self.lr = lr

        self.lstms = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_units *
            self.pred_window,
            batch_first=True,
            num_layers=lstm_layers)
        self.dense = nn.Linear(
            in_features=self.window *
            self.hidden_units *
            self.pred_window,
            out_features=self.hidden_units *
            self.pred_window)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, hidden = self.lstms(x)
        x = x.reshape(-1, self.window * self.hidden_units * self.pred_window)
        x = self.dense(x).reshape(-1, self.pred_window, self.hidden_units)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        if np.any(y.shape != y_hat.shape):
            logger.warning(
                "The shape output from the model is expected "
                "to be [1, # of columns]")
            y = y.reshape(*y_hat.shape)
        loss = self.criterion(y_hat, y, reduction="mean")
        metrics = {'train_loss': loss, "#batches": self.trainer.num_training_batches}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        X, y = batch
        y_hat = self(X)
        anomaly_score = self.criterion(
            y_hat.detach(),
            y.detach(),
            reduction="none"
        ).sum(dim=[1])
        return anomaly_score

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
