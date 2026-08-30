import torch

def tanh_forward_backward(x, upstream_gradient):
    """
    Returns: tanh output and its upstream-scaled input gradient
    """
    pass
    y = torch.tanh(x)
    return (y, upstream_gradient * (1 - y ** 2))