import pytorch_lightning as pl
from torch import nn
import torch
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union


def l2_norm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.norm(x - y, dim=1)


def l1_norm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.norm(x - y, p=1, dim=1)


class ForecastingModule(pl.LightningModule):
    """Forecastring model for anomaly detection.
    """
    model: nn.Module()

    def __init__(self,
                 model: nn.Module,
                 lr: float,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam):
        super().__init__()
        self.lr = lr
        self.optimizer_class = optimizer_class
        self.criterion = nn.MSELoss(reduction='mean')
        self.anomaly_criterion = l2_norm
        self.model = model
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self,
                     batch,
                     batch_idx,
                     dataloader_idx=None) -> Tuple[torch.Tensor,
                                                   torch.Tensor,
                                                   torch.Tensor]:
        X, y, x_index, y_index = batch
        y_hat = self(X)
        return y_hat, self.anomaly_criterion(y_hat, y), y_index

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
