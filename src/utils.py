"""Utility functions for the project."""


def network_parameters(nets):
    """Calculate the total number of parameters in a network.

    Args:
        nets: A PyTorch neural network model

    Returns:
        int: Total number of parameters in the network
    """
    num_params = sum(param.numel() for param in nets.parameters())
    return num_params
