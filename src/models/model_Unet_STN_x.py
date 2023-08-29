import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import model_utils
from bilinear_interpolation import BilinearInterpolation


class STN(nn.Module):
    def __init__(self, input_shape=(1, 32, 64), sampling_size=(8, 16), num_classes=10):
        super(STN, self).__init__()
        self.input_shape = input_shape
        self.sampling_size = sampling_size
        self.num_classes = num_classes

        # Note: PyTorch uses B, C, H, W ordering while TensorFlow uses B, H, W, C
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.locnet = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * self.sampling_size[0] * self.sampling_size[1], 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            # Initialize weights here if necessary_
            nn.Linear(50, 6),
        )  # The six-dimensional torch is the radial transformation parameter

        # Initialize the weights of the last Linear layer
        (
            self.locnet[-1].weight.data,
            self.locnet[-1].bias.data,
        ) = model_utils.get_initial_weights_torch(50)

        self.bilinear_interpolation = BilinearInterpolation(self.sampling_size)

        self.upconv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=2, padding=1), nn.ReLU()  # up6
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.upconv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=2, padding=1), nn.ReLU()  # up7
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.conv10 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x5 = self.conv5(x3)

        # Apply locnet to flattened x5
        theta = self.locnet(x5)

        # Use bilinear interpolation on x with `theta`
        x_transformed = self.bilinear_interpolation(
            x, theta
        )  # x is the input image, theta is the transformation parameter

        up6 = F.interpolate(x_transformed, scale_factor=2, mode="nearest")
        up6 = self.upconv1(up6)
        up6 = torch.cat([up6, x2], 1)
        x6 = self.conv6(up6)

        up7 = F.interpolate(x6, scale_factor=2, mode="nearest")
        up7 = self.upconv2(up7)
        up7 = torch.cat([up7, x1], 1)
        x7 = self.conv7(up7)

        x10 = self.conv10(x7)
        return x10


model = STN()
print(model)
