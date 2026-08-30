import numpy as np

def gradient_descent_step(values, gradients, learning_rate):
    """
    Returns: updated values and the predicted first-order objective change
    """
    values = np.array(values, dtype=np.float64)
    gradients = np.array(gradients, dtype=np.float64)

    updated_values = values - learning_rate * gradients
    delta_l_pred = np.sum(gradients @ (updated_values - values))

    return (list(updated_values), float(delta_l_pred))