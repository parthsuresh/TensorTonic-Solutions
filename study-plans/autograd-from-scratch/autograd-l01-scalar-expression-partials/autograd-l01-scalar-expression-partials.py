import numpy as np

def scalar_expression_partials(a, b, c, h):
    """
    Returns: the expression value and its three numerical partial derivatives
    """
    d = a * b + c
    d_a = (((a + h) * b + c) - d) / h
    d_b = ((a * (b +h) + c) - d) / h
    d_c = ((a * b + (c + h)) - d) / h
    return (d, d_a, d_b, d_c)
    