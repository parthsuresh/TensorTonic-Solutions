import numpy as np

def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    k = len(coefficients)
    coefficients = np.array(coefficients, dtype=np.float64)

    def horner(c, xv):
        acc = 0.0
        for coef in c[::-1]:
            acc = acc * xv + coef
        return acc

    f_x = horner(coefficients, x)
    f_x_plus_h = horner(coefficients, x + h)
    estimated_slope = (f_x_plus_h - f_x) / h
    return (f_x, f_x_plus_h, estimated_slope)