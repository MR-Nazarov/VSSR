import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple


class DnCNN(nn.Module):
    def __init__(
        self, 
        num_layers: int = 17,
        n_channels: int = 64, 
        image_channels: int = 1,
        use_bnorm: bool = True,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        # Handle flexible kernel size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = tuple(k // 2 for k in kernel_size)
        
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
        layers = []
        
        # First layer (with bias, like original)
        layers.append(nn.Conv2d(
            in_channels=image_channels, 
            out_channels=n_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            bias=True
        ))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(
                in_channels=n_channels, 
                out_channels=n_channels, 
                kernel_size=kernel_size, 
                padding=padding, 
                bias=not use_bnorm  # No bias when using batch norm
            ))
            
            if use_bnorm:
                layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            
            layers.append(nn.ReLU(inplace=True))
            
            # Add dropout after ReLU if specified
            if self.dropout is not None:
                layers.append(self.dropout)
        
        # Last layer (no bias, like original)
        layers.append(nn.Conv2d(
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
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.dncnn(x)
        return x - noise  # Residual learning: clean_image = noisy_image - predicted_noise