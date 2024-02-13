import pytorch_lightning as pl
from torch import nn
import torch
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union
import logging
logger = logging.getLogger(__name__)


def l2_norm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.norm(x - y, dim=1)


def l1_norm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.norm(x - y, p=1, dim=1)


class ReconstructionModule(pl.LightningModule):
    """Reconstruction model for anomaly detection.
    """

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
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        The training step is called for every batch.
        batch (Tuple[torch.Tensor, torch.Tensor]):
        A tuple containing the input and the target.
            - target is the input itself in the case of an autoencoder.
        """
        x, y = batch  # Assuming the dataset returns the input and the target.
        y_hat = self(x)
        try:
            loss = self.criterion(y_hat, y)
        except Exception:
            logger.exception("Error in the training step"
                             "This probably happnes when the input "
                             "and the target have different shapes"
                             )
            if (y_hat.shape == x.shape):
                logger.warning("Shapes of x and y_hat are the same,"
                               "This might be a problem of the dataloader"
                               " and dataset implementation."
                               " Note for autoencoder, the target "
                               "should be the input itself in "
                               "most cases.")

            raise

        metrics = {'train_loss': loss, "#batches": self.trainer.num_training_batches}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
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
