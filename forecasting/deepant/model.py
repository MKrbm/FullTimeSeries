
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch import nn
import logging
logger = logging.getLogger(__name__)


class DeepAnT(pl.LightningModule):
    def __init__(
            self,
            window,
            pred_window,
            in_channels,
            filter1_size,
            filter2_size,
            kernel_size,
            pool_size,
            stride):
        super(DeepAnT, self).__init__()
        self.window = window
        self.pred_window = pred_window
        self.in_channels = in_channels
        self.filter1_size = filter1_size
        self.filter2_size = filter2_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.stride = stride
        self.criterion = nn.functional.mse_loss

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=filter1_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=0)
        self.conv2 = nn.Conv1d(
            in_channels=filter1_size,
            out_channels=filter2_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=0)
        self.maxpool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(0.25)
        # Ensure this calculation is correct for your architecture
        self.dim1 = int(0.5*(0.5*(window-1)-1)) * filter2_size
        self.lin1 = nn.Linear(self.dim1, in_channels*pred_window)
        self.save_hyperparameters()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, self.dim1)
        x = self.dropout(x)
        x = self.lin1(x)
        return x.view(-1, self.pred_window, self.in_channels)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer
