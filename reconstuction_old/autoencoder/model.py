import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import logging
logger = logging.getLogger(__name__)


class PointAutoencoder(nn.Module):
    """
    A point autoencoder model.

    The shape of input dataset is (n_bathces, window, feature), 
    but the linear layers is only applied to the last layer.
    Args:
        input_size (int): The number of input features. 
        latent_dim (int): The number of latent dimensions.
        learning_rate (float): The learning rate for the optimizer.
    """

    def __init__(self, input_size, latent_dim=32):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_size, latent_dim),
            nn.Relu(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Relu(),
            nn.Linear(latent_dim, input_size),
        )

    def forward(self, x):
        print(x.shape)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder(nn.Module):
    """
    A simple autoencoder model for 3D tensors.

    Args:
        window_size (int): The window size.
        feature_size (int): The number of input features per timestep.
        latent_dim (int): The size of the latent dimension.

    The model expects input data in the shape (n_batches, window, feature).
    It first reshapes (flattens) the 3D tensor into a 2D tensor of shape (n_batches, window * feature),
    processes it through the encoder and decoder, and finally reshapes it back to the original 3D shape.
    """

    def __init__(self, window_size, feature_size, latent_dim=32):
        super().__init__()
        self.window_size = window_size
        self.feature_size = feature_size
        self.latent_dim = latent_dim
        input_size = window_size * feature_size  # Calculate flattened input size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_size),
        )

    def forward(self, x):
        # Reshape (flatten) the input to 2D
        x = x.reshape(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Reshape the output back to 3D
        decoded = decoded.reshape(
            x.size(0), self.window_size, self.feature_size)
        return decoded
