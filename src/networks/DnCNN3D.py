import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple


class DnCNN3D(nn.Module):
    def __init__(
        self,
        num_layers: int = 17,
        n_channels: int = 64,
        image_channels: int = 1,
        use_bnorm: bool = False,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        dropout_rate: float = 0.0
    ):
        """
        3D DnCNN network for volumetric denoising.

        Args:
            num_layers: Number of convolutional layers (default: 17)
            n_channels: Number of feature channels in hidden layers (default: 64)
            image_channels: Number of input/output channels (default: 1)
            use_bnorm: Whether to use batch normalization (default: True)
            kernel_size: Size of convolutional kernels (default: 3)
            dropout_rate: Dropout rate (default: 0.0, no dropout)
        """
        super().__init__()

        # Handle flexible kernel size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        padding = tuple(k // 2 for k in kernel_size)

        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else None

        layers = []

        # First layer (with bias, like original)
        layers.append(nn.Conv3d(
            in_channels=image_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        ))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv3d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=not use_bnorm  # No bias when using batch norm
            ))

            if use_bnorm:
                layers.append(nn.BatchNorm3d(n_channels, eps=0.0001, momentum=0.95))

            layers.append(nn.ReLU(inplace=True))

            # Add dropout after ReLU if specified
            if self.dropout is not None:
                layers.append(self.dropout)

        # Last layer (no bias, like original)
        layers.append(nn.Conv3d(
            in_channels=n_channels,
            out_channels=image_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        ))

        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using residual learning.

        Args:
            x: Input noisy volume of shape (B, C, D, H, W)

        Returns:
            Denoised volume of shape (B, C, D, H, W)
        """
        noise = self.dncnn(x)
        return x - noise  # Residual learning: clean_volume = noisy_volume - predicted_noise
