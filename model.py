
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearEncoder(nn.Module):
    """Single fully connected linear encoder."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LinearDecoder(nn.Module):
    """Single fully connected linear decoder."""

    def __init__(self, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class Autoencoder(nn.Module):
    """Single-layer linear autoencoder."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = LinearEncoder(input_dim, latent_dim)
        self.decoder = LinearDecoder(latent_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent


class LatentDiscriminator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.10,
        n_conditions: int = 0,
    ) -> None:
        super().__init__()
        self.n_conditions = int(n_conditions)

        self.network = nn.Sequential(
            nn.Linear(latent_dim + self.n_conditions, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        latent: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.n_conditions > 0:
            if condition is None:
                raise ValueError("Condition labels are required.")
            one_hot = F.one_hot(
                condition.long(),
                num_classes=self.n_conditions,
            ).float()
            latent = torch.cat([latent, one_hot], dim=1)

        return self.network(latent)


def initialize_weights(module: nn.Module) -> None:

    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def freeze_module(module: nn.Module) -> None:

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
