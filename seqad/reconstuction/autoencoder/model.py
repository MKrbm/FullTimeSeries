import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import logging
logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """
    A simple autoencoder model.

    Args:
        input_size (int): The number of input features.
        latent_dim (int): The number of latent dimensions.
        learning_rate (float): The learning rate for the optimizer.
    """
    def __init__(self, input_size, latent_dim=32):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

