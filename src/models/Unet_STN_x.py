import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self, input_shape=(1, 32, 64), sampling_size=(8, 16), num_classes=10):
        super(STN, self).__init__()

        # Localisation network
        self.localization = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            # Fully connected layer
            nn.Flatten(),
            nn.Linear(32 * 8 * 16, 500),
            nn.ReLU(True),
            nn.Linear(500, 200),
            nn.ReLU(True),
            nn.Linear(200, 100),
            nn.ReLU(True),
            nn.Linear(100, 50),
            nn.ReLU(True),
            nn.Linear(50, 6),
        )

        # Initialize the weights/bias with identity transformation
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

        # Rest of the network
        # ... to be added ...

    def forward(self, x):
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x  # Continue the network operations...


# Instantiate and print the network to verify
stn = STN()
print(stn)
