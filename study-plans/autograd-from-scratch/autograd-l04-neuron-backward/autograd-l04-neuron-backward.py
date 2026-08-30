import torch

def neuron_backward(inputs, weights, bias, upstream_gradient):
    """
    Returns: output, input gradients, weight gradients, and bias gradient
    """
    inputs = torch.tensor(inputs)
    weights = torch.tensor(weights)

    a = inputs @ weights + bias
    delta =upstream_gradient * (1 - torch.tanh(a) ** 2)
    
    dL_dx = delta * weights
    dL_dw = delta * inputs
    dL_db = delta

    return (torch.tanh(a), dL_dx, dL_dw, dL_db)
