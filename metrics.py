import torch
import torch.nn as nn
import hydra
from collections import OrderedDict


class MetricCollection(nn.Module):
    """Custom metric collection that works with Hydra"""

    def __init__(self, **metrics):
        super().__init__()
        self.metrics = nn.ModuleDict()

        for name, metric_config in metrics.items():
            if hasattr(metric_config, '_target_'):
                # This is a Hydra config, instantiate it
                self.metrics[name] = hydra.utils.instantiate(metric_config)
            else:
                # This is already an instantiated metric
                self.metrics[name] = metric_config

    def __call__(self, pred, target):
        """Compute all metrics"""
        results = {}
        for name, metric in self.metrics.items():
            try:
                results[name] = metric(pred, target)
            except Exception as e:
                print(f"Error computing {name}: {e}")
                results[name] = torch.tensor(float('nan'))
        return results

    def update(self, pred, target):
        """Update all metrics (for metrics that accumulate)"""
        for name, metric in self.metrics.items():
            if hasattr(metric, 'update'):
                try:
                    metric.update(pred, target)
                except Exception as e:
                    print(f"Error updating {name}: {e}")

    def compute(self):
        """Compute final values for all metrics"""
        results = {}
        for name, metric in self.metrics.items():
            if hasattr(metric, 'compute'):
                try:
                    results[name] = metric.compute()
                except Exception as e:
                    print(f"Error computing final {name}: {e}")
                    results[name] = torch.tensor(float('nan'))
            else:
                results[name] = torch.tensor(float('nan'))
        return results

    def reset(self):
        """Reset all metrics"""
        for metric in self.metrics.values():
            if hasattr(metric, 'reset'):
                metric.reset()

    def to(self, device):
        """Move all metrics to device"""
        super().to(device)
        for metric in self.metrics.values():
            if hasattr(metric, 'to'):
                metric.to(device)
        return self