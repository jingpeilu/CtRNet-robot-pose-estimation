import os
import cv2
import sys
import numpy as np
import torch
import torch.nn.functional as F


def heatmap_to_keypoints(heatmaps, temperature=1e-6):
    """
    Convert heatmaps to keypoints using soft-argmax with a very small temperature parameter.
    
    Args:
        heatmaps (torch.Tensor): Heatmaps of shape (batch_size, num_keypoints, height, width)
        temperature (float): Temperature parameter for the softmax function. A very small value makes the distribution sharper.
        
    Returns:
        torch.Tensor: Keypoints of shape (batch_size, num_keypoints, 2)
    """
    assert heatmaps.dim() == 4, "Heatmaps should be a 4D tensor (batch_size, num_keypoints, height, width)"
    
    # Get the batch size, number of keypoints, height, and width
    batch_size, num_keypoints, height, width = heatmaps.size()
    
    # Flatten the heatmaps to (batch_size, num_keypoints, height * width)
    heatmaps_flat = heatmaps.view(batch_size, num_keypoints, -1)
    
    # Apply softmax with a small temperature to approximate argmax
    heatmaps_flat = F.softmax(heatmaps_flat / temperature, dim=-1)
    
    # Reshape back to (batch_size, num_keypoints, height, width)
    heatmaps = heatmaps_flat.view(batch_size, num_keypoints, height, width)
    
    # Create coordinate grids
    y_coords = torch.arange(height, dtype=torch.float32, device=heatmaps.device).view(1, 1, height, 1)
    x_coords = torch.arange(width, dtype=torch.float32, device=heatmaps.device).view(1, 1, 1, width)
    
    # Compute the expected coordinates using the softmax probabilities as weights
    keypoints_y = torch.sum(heatmaps * y_coords, dim=[2, 3])
    keypoints_x = torch.sum(heatmaps * x_coords, dim=[2, 3])
    
    # Stack the coordinates to get (x, y) pairs
    keypoints = torch.stack((keypoints_x, keypoints_y), dim=2)
    
    return keypoints

def compute_gaussian_heatmap(coords:torch.Tensor, size:torch.Size, std:float = 1.5) -> torch.Tensor:
    '''
    Given an Nx2 tensor of coordinates, compute the gaussian heatmap (1 channel per keypoint)
    This is basically a wrapper for render_gaussian2d
    
    Inputs:
        coords: (B,N,2) or (N,2) tensor of coordinates
        size: Size of the output score map (H, W)
        std: Standard deviation of the gaussian kernel
    
    Outputs:
        heatmap: (B,N,H,W) or (N, H, W) tensor of gaussian heatmaps
    '''
    assert coords.dim() == 2 or coords.dim() == 3, "Coordinates must be 2D or 3D tensor"
    assert std > 0, "Standard deviation must be positive"
    assert len(size) >= 2, "Size must be at least 2D and the last 2 dimensions are used for the heatmap"

    # Convert coords to 2D tensor if necessary
    batch_size = None
    if coords.dim() == 3:
        batch_size = coords.shape[0]
        coords = coords.flatten(0, 1)

    # Initialize gaussian heatmap as zeros (N, H, W)
    heatmap = torch.zeros((coords.shape[0], size[-2], size[-1]), device=coords.device)

    # Create grid of coordinates
    x = torch.arange(size[-2], device=coords.device)
    y = torch.arange(size[-1], device=coords.device)
    x_grid, y_grid = torch.meshgrid(x, y)

    # Create gaussian heatmap for each non-NaN landmark
    for i in range(coords.shape[0]):
        if not torch.isnan(coords[i, 0]) and not torch.isnan(coords[i, 1]):
            # Compute gaussian heatmap
            heatmap[i] = torch.exp(-((x_grid - coords[i, 1])**2 + (y_grid - coords[i, 0])**2) / (2 * std**2))
            if heatmap[i].max() > 1e-6:
                heatmap[i] = heatmap[i] / heatmap[i].max()

    # Reshape heatmap to (B,N,H,W) if necessary
    if batch_size is not None:
        heatmap = heatmap.reshape(batch_size, -1, size[-2], size[-1])

    return heatmap


class GaussianHeatmapLoss(torch.nn.Module):
    def __init__(self, std:float = 1.5, landmark_downscale:float = 1):
        '''
            L2 loss for gaussian heatmaps
            Inputs:
                std: (float) standard deviation of the gaussian kernel
                landmark_downscale: (float) downscale factor of the landmarks
        '''
        super(GaussianHeatmapLoss, self).__init__()
        self.std = std
        self.mse = torch.nn.MSELoss()
        self.landmark_downscale = landmark_downscale
    
    def forward(self, source:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        '''
            Loss function is:
                loss = MSE(score_map, heatmap(target) )
            
            Inputs:
                source: (dict) output of model.forward()
                        'score_map': (B, num_keypoints, H, W) tensor. The score map for each keypoint.
                target: (B, num_keypoints, 2) tensor. The target coordinates for each keypoint.
        '''
        assert isinstance(source, torch.Tensor), "source must be a tensor"
        assert isinstance(target, torch.Tensor), "target must be a tensor"
        assert len(source.shape) == 4, "source must have 4 dimensions"
        assert len(target.shape) == 3, "target must have 4 dimensions"
        assert source.shape[0] == target.shape[0], "Batch size of source and target must be the same"
        assert source.shape[1] == target.shape[1], "Number of keypoints of source and target must be the same"
        assert target.shape[2] == 2, "target must have 2 dimensions for the keypoint location"

        # If landmark_downscale is not 1, then we need to downscale the target
        if self.landmark_downscale != 1:
            target = target / self.landmark_downscale

        # Create the gaussian heatmaps
        #   - Reshape target to (B*num_keypoints, 2) so that we can compute the gaussian heatmap for each keypoint
        #   - Compute gaussian heatmap
        target_heatmap = compute_gaussian_heatmap(target.reshape(-1, 2), source.shape[2:], self.std)
        #   - Reshape target_heatmap to (B, num_keypoints, H, W)
        target_heatmap = target_heatmap.reshape(source.shape[0], source.shape[1],
                                                source.shape[2], source.shape[3])

        # Compute loss
        loss = self.mse(source, target_heatmap)

        return loss, target_heatmap

    def __str__(self):
        return f"Gaussian Heatmap Loss with std={self.std} and landmark_downscale={self.landmark_downscale}"

    def __repr__(self):
        return f"Gaussian Heatmap Loss with std={self.std} and landmark_downscale={self.landmark_downscale}"