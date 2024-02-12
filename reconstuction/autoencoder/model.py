import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import logging
logger = logging.getLogger(__name__)


class Autoencoder(pl.LightningModule):
    """
    A simple autoencoder model.

    Args:
        input_size (int): The number of input features.
        latent_dim (int): The number of latent dimensions.
        learning_rate (float): The learning rate for the optimizer.
    """
    def __init__(self, input_size, latent_dim=32, lr=0.005):
        super(Autoencoder, self).__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_size)
        )
        self.criterion = nn.functional.mse_loss
        self.save_hyperparameters()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        """
        The training step is called for every batch.
        batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the input and the target.
            - target is the input itself in the case of an autoencoder.
        """
        x, _ = batch  # Assuming the dataset returns the input and the target.
        x_hat = self(x)
        loss = self.criterion(x_hat, x, reduction='mean')
        metrics = {'train_loss': loss, "#batches": self.trainer.num_training_batches}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        x_hat = self(x)
        loss = self.criterion(x_hat, x, reduction='none').sum(dim=[1])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
