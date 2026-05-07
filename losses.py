import torch
import torch.nn as nn
import hydra

class CombinedLoss(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = nn.ModuleDict()
        for name, loss_config in losses.items():
            self.losses[name] = hydra.utils.instantiate(loss_config)
        self.weights = weights

    def forward(self, pred, target):
        total_loss = 0
        loss_dict = {}

        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(pred, target)
            weighted_loss = self.weights[name] * loss_value
            total_loss += weighted_loss
            loss_dict[f'{name}_loss'] = loss_value.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict


