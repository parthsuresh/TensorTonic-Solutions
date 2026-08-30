import torch

def neuron_forward(inputs, weights, bias):
    """
    Returns: scalar preactivation and tanh output
    """
    inputs = torch.tensor(inputs)
    weights = torch.tensor(weights)
    a = inputs @ weights + bias
    return (a, torch.tanh(a))