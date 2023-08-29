import numpy as np
import torch


def get_initial_weights_torch(output_size):
    b = torch.zeros(2, 3, dtype=torch.float32)
    # Identity transformation: set the main diagonal to 1
    b[0, 0] = 1
    b[1, 1] = 1

    # Initialize the weights to zero
    W = np.zeros((output_size, 6), dtype="float32")
    W = torch.tensor(W, dtype=torch.float)
    # b = torch.tensor(b.flatten(), dtype=torch.float)
    b = torch.as_tensor(b.flatten(), dtype=torch.float) # Don't compute gradient for b when initializing weights

    return (W, b)


# The function implementation below is a modification based on DDWP-DA's GitHub code
# Original code link: https://github.com/ashesh6810/DDWP-DA/blob/master/utils.py
def get_initial_weights(output_size):
    b = np.zeros((2, 3), dtype="float32")
    # Identity transformation: set the main diagonal to 1
    b[0, 0] = 1
    b[1, 1] = 1
    # Initialize the weights to zero
    W = np.zeros((output_size, 6), dtype="float32")
    # Put weight W and bias b (after flattening) into a list
    weights = [W, b.flatten()]
    return weights
