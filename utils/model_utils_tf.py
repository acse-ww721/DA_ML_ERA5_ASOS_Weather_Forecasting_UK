# Name: Wenqi Wang
# Github username: acse-ww721

import numpy as np

# The function implementation below is a modification based on DDWP-DA's GitHub code
# Original code link: https://github.com/ashesh6810/DDWP-DA/blob/master/utils.py


def get_initial_weights(output_size):
    """
    Initialize weights for a neural network layer.

    Args:
        output_size (int): The number of output units for the layer.

    Returns:
        list: A list containing two numpy arrays - the weight matrix (W) and the bias vector (b).

    Example:
        >>> get_initial_weights(4)
        [array([[0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.]]), array([1., 0., 0., 1., 0., 0.], dtype=float32)]
    """
    b = np.zeros((2, 3), dtype="float32")
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((output_size, 6), dtype="float32")
    weights = [W, b.flatten()]
    return weights
