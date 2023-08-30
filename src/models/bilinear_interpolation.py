import torch
import torch.nn as nn
import torch.nn.functional as F

# The function implementation below is a modification version from Tensorflow to Pytorch
# Original code link: https://github.com/ashesh6810/DDWP-DA/blob/master/layers.py


class BilinearInterpolation(nn.Module):
    """Performs bilinear interpolation as a PyTorch layer.
    References:
    ----------
    [1] Spatial Transformer Networks, Max Jaderberg, et al.
    [2] https://github.com/skaae/transformer_network
    [3] https://github.com/EderSantana/seya
    """

    def __init__(self, output_size, device='cpu'):
        super(BilinearInterpolation, self).__init__()
        self.output_size = output_size
        self.device = device

    def forward(self, X, affine_transformation):
        transformed = self._transform(X, affine_transformation)
        return transformed

    def _interpolate(self, image, sampled_grids):
        # [B, H, W, C] this should be converted to [B, C, H, W] first to apply pytorch format
        # Convert [B, H, W, C] to [B, C, H, W]
        image = image.permute(0, 3, 1, 2)
        batch_size, num_channels, height, width = image.shape

        x = (sampled_grids[:, 0, :] + 1) * width * 0.5
        y = (sampled_grids[:, 1, :] + 1) * height * 0.5

        # Convert to integer part and floating part
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        # Ensure within bounds
        max_x = width - 1
        max_y = height - 1
        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        flat_image = image.reshape(batch_size, num_channels, -1)

        # Handle batch indices
        pixels_batch = torch.arange(batch_size, device=image.device).unsqueeze(-1) * (
            height * width
        )
        base = pixels_batch.expand(-1, sampled_grids.shape[2])

        base_y0 = base + y0 * width
        base_y1 = base + y1 * width

        # Calculate indices for 4 corners
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        # Look up pixel values
        pixel_values_a = torch.gather(
            flat_image, 2, indices_a.unsqueeze(-1).expand(-1, num_channels, -1)
        )
        pixel_values_b = torch.gather(
            flat_image, 2, indices_b.unsqueeze(-1).expand(-1, num_channels, -1)
        )
        pixel_values_c = torch.gather(
            flat_image, 2, indices_c.unsqueeze(-1).expand(-1, num_channels, -1)
        )
        pixel_values_d = torch.gather(
            flat_image, 2, indices_d.unsqueeze(-1).expand(-1, num_channels, -1)
        )

        x0, x1, y0, y1 = x0.float(), x1.float(), y0.float(), y1.float()
        area_a = ((x1 - x) * (y1 - y)).unsqueeze(-1)
        area_b = ((x1 - x) * (y - y0)).unsqueeze(-1)
        area_c = ((x - x0) * (y1 - y)).unsqueeze(-1)
        area_d = ((x - x0) * (y - y0)).unsqueeze(-1)

        interpolated_image = (
            area_a * pixel_values_a
            + area_b * pixel_values_b
            + area_c * pixel_values_c
            + area_d * pixel_values_d
        )

        return interpolated_image  # [B, C, H, W]
        # return interpolated_image.permute(0, 3, 1, 2) # [B, H, W, C]

    def _make_regular_grids(self, batch_size, height, width):
        """
        First generate uniform values for width and height in the range -1 to 1.
        Next, we create a grid using torch.meshgrid, then concatenate the x and y coordinates
        and add an array of all ones.
        Finally, we repeat this grid so that it matches the batch size.
        """
        x_linspace = torch.linspace(-1.0, 1.0, width).float().to(self.device)
        y_linspace = torch.linspace(-1.0, 1.0, height).float().to(self.device)
        # x_linspace = torch.linspace(-1.0, 1.0, width).float().to(image.device) # GPU
        # y_linspace = torch.linspace(-1.0, 1.0, height).float().to(image.device) # GPU
        x_coordinates, y_coordinates = torch.meshgrid(x_linspace, y_linspace)
        x_coordinates = x_coordinates.flatten(0)
        y_coordinates = y_coordinates.flatten(0)
        ones = torch.ones_like(x_coordinates)
        grid = torch.stack([x_coordinates, y_coordinates, ones], 0)
        grid = grid.repeat(batch_size, 1, 1)
        return grid

    def _transform(self, X, affine_transformation):
        batch_size, _, _, _ = X.shape
        theta = affine_transformation.view(batch_size, 2, 3)  # [B, 2, 3]
        regular_grids = self._make_regular_grids(batch_size, *self.output_size)
        sampled_grids = torch.bmm(theta, regular_grids)
        transformed_image = self._interpolate(X, sampled_grids)
        # Convert [B, C, H, W] back to [B, H, W, C]
        transformed_image = transformed_image.permute(0, 2, 3, 1)

        return transformed_image


#  Output size not may not match
