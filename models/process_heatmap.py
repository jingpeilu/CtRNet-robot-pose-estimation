import torch
import torch.nn as nn

from typing import List, Optional, Tuple

def _get_spatial_gradient_kernel(order: int, dtype: torch.dtype) -> torch.Tensor:
    '''
    Get the spatial gradient kernel for the first or second order derivative.
    Args:
        order (int): the order of the derivative. Only 1 and 2 are supported.
        dtype (torch.dtype): data type of the weights
    Returns:
        torch.Tensor: the kernel with shape :math:`(2, 5, 5)` for the first order derivative (dx, dy)
                        or shape :math:`(3, 5, 5)` for the second order derivative (dxx, dyy, dxy)

        Original implementation for 2nd order derivative where spacing of 2 is used
        https://github.com/ilovepose/DarkPose/blob/0185b427f34a08d93ea7e948ccc3e7fdc78d88c5/lib/core/inference.py#L51
        Note ViTPose uses a spacing of 1
        https://github.com/ViTAE-Transformer/ViTPose/blob/d5216452796c90c6bc29f5c5ec0bdba94366768a/mmpose/core/evaluation/top_down_eval.py#L335
    
        Instead, I am going to use optimal spacing from this tool and always use a kernel size of 5
        https://web.media.mit.edu/~crtaylor/calculator.html

    '''
    if order == 1:
        kernel_x = torch.zeros((5, 5),  dtype=dtype)
        kernel_x[2, :] = torch.tensor([1./12., -8/12., 0, 8/12, -1./12], dtype=dtype)
        kernel_y = kernel_x.transpose(0, 1)
        kernel = torch.stack([kernel_x, kernel_y], dim=0)
    elif order == 2:
        kernel_xx = torch.zeros((5, 5),  dtype=dtype)
        kernel_xx[2, :] = torch.tensor([-1./12., 16/12., -30./12., 16/12, -1./12], dtype=dtype)
        kernel_yy = kernel_xx.transpose(0, 1)

        kernel_xy = torch.zeros((5, 5), dtype=dtype)
        kernel_xy[0,0] = -1./48 # (x-2, y-2)
        kernel_xy[1,1] =  16./48 # (x-1, y-1)
        kernel_xy[0,4] =  1./48 # (x-2, y+2)
        kernel_xy[1,3] = -16./48 # (x-1, y+1)
        kernel_xy[4, 0] =  1./48 # (x+2, y-2)
        kernel_xy[3, 1] = -16./48 # (x+1, y-1)
        kernel_xy[4, 4] = -1./48 # (x+2, y+2)
        kernel_xy[3, 3] =  16./48 # (x+1, y+1)

        kernel = torch.stack([kernel_xx, kernel_yy, kernel_xy], dim=0)
    else:
        raise NotImplementedError("Only derivatives up to order 2nd order are supported.")

    return kernel

def _get_gaussian_kernel1d(kernel_size: int, sigma: float, dtype: torch.dtype) -> torch.Tensor:
    '''
    Get the 1D gaussian kernel.
    Args:
        kernel_size (int): the size of the kernel.
        sigma (float): the standard deviation of the gaussian kernel.
        dtype (torch.dtype): data type of the weights
    Returns:
        torch.Tensor: the 1D gaussian kernel.
    
    '''
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).to(dtype=dtype)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()
    return kernel1d

def _get_gaussian_kernel2d(kernel_size: List[int], sigma: List[float], dtype: torch.dtype) -> torch.Tensor:
    '''
    Get the 2D gaussian kernel.
    Args:
        kernel_size (List[int]): the size of the kernel. 
            Should be a list of 2 integers (for x and y dimensions)
        sigma (List[float]): the standard deviation of the gaussian kernel. 
            Should be a list of 2 floats (for x and y dimensions)
    Returns:
        torch.Tensor: the 2D gaussian kernel.
    '''
    assert len(kernel_size) == 2, "Kernel size should be a list of 2 integers"
    assert len(sigma) == 2, "Sigma should be a list of 2 floats"

    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0], dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1], dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d

class ImageDerivative(nn.Module):
    def __init__(self, num_channels:int, dtype:torch.dtype = torch.float32):
        '''
            Module that computes the spatial derivative and hessians of the image using a 5x5 kernel
            For more details on the kernel, see _get_spatial_gradient_kernel
        '''

        super(ImageDerivative, self).__init__()

        # Create spatial gradient layer for first and second derivatives. Order for output channels is:
        #       [dx_1 , dy_1  , dxx_1  , dyy_1  , dxy_1, ..., dx_N , dy_N  , dxx_N  , dyy_N  , dxy_N]
        self.derivative = nn.Conv2d(num_channels, 5*num_channels, kernel_size=5,
                                    padding=2, bias=False, padding_mode='reflect', groups=num_channels)

        # Initialize weights of the spatial gradient layer
        self.derivative.weight.data[::5]  = _get_spatial_gradient_kernel(1, dtype=dtype)[0].unsqueeze(0).unsqueeze(0).expand(num_channels, 1, -1, -1)
        self.derivative.weight.data[1::5] = _get_spatial_gradient_kernel(1, dtype=dtype)[1].unsqueeze(0).unsqueeze(0).expand(num_channels, 1, -1, -1)        
        self.derivative.weight.data[2::5] = _get_spatial_gradient_kernel(2, dtype=dtype)[0].unsqueeze(0).unsqueeze(0).expand(num_channels, 1, -1, -1)
        self.derivative.weight.data[3::5] = _get_spatial_gradient_kernel(2, dtype=dtype)[1].unsqueeze(0).unsqueeze(0).expand(num_channels, 1, -1, -1)
        self.derivative.weight.data[4::5] = _get_spatial_gradient_kernel(2, dtype=dtype)[2].unsqueeze(0).unsqueeze(0).expand(num_channels, 1, -1, -1)

        # Freeze the weights
        for param in self.derivative.parameters():
            param.requires_grad = False

    def forward(self, image:torch.Tensor) -> torch.Tensor:
        '''
        Computes the spatial derivative and hessians of the image using a 5x5 kernel
        Args:
            image (torch.Tensor): image of shape (batch_size, num_channels, height, width)
        Returns:
            image_der (torch.Tensor): tensor of shape (batch_size, 5*num_channels, height, width)
                where the order of the channel dimension is 
                [dx_1 , dy_1  , dxx_1  , dyy_1  , dxy_1, ..., dx_N , dy_N  , dxx_N  , dyy_N  , dxy_N]
        '''
        return self.derivative(image)

class GaussianBlurImage(nn.Module):
    def __init__(self, num_channels:int, sigma:float, dtype:torch.dtype = torch.float32):
        '''
        Module that performs gaussian blur on the image
        Args:
            num_channels (int): number of channels in the image
            sigma (float): sigma for gaussian blur
            dtype (torch.dtype): data type of the weights
        '''
        super(GaussianBlurImage, self).__init__()

        # Create conv2d for gaussian blur
        kernel_size = round( ( sigma - 0.8 ) / 0.3 + 1 ) * 2 + 1
        padding = kernel_size // 2
        self.blur = nn.Conv2d(num_channels, num_channels, kernel_size, padding=padding, bias=False, groups=num_channels)

        # Initialize weights of the blur layer
        self.blur.weight.data = _get_gaussian_kernel2d([kernel_size, kernel_size], 
                                                        [sigma, sigma],
                                                        dtype=dtype).unsqueeze(0).unsqueeze(0).expand(num_channels, -1, -1, -1)
        
        # Freeze the weights
        for param in self.blur.parameters():
            param.requires_grad = False

    def forward(self, image:torch.Tensor) -> torch.Tensor:
        '''
        Performs gaussian blur on the image
        Args:
            image (torch.Tensor): image of shape (batch_size, num_channels, height, width)
        Returns:
            image_blur (torch.Tensor): image of shape (batch_size, num_channels, height, width)
        '''
        return self.blur(image)

class MaxLandmarkDARK(nn.Module):
    def __init__(self, num_channels:int, sigma:Optional[float] = None, dtype:torch.dtype = torch.float32,
                 use_gather:bool = True, clamp_output:bool = True):
        '''
        Finds the location and value of the maximum value in each heatmap channel using the DARK method
        Args:
            num_channels (int): number of channels in the heatmap
            sigma (float): sigma for gaussian blur
            dtype (torch.dtype): data type of the weights
            use_gather (bool): whether to use torch.gather to get the gradient and hessian at the max location
                Turning off is useful when exporting the model and torch.gather is not supported
            clamp_output (bool): whether to clamp the output to be within the heatmap
                Turning off is useful when exporting the model and torch.clamp is not supported
        '''
        super(MaxLandmarkDARK, self).__init__()

        # Create conv2d for gaussian blur if sigma is not None
        if sigma is not None:
            self.blur = GaussianBlurImage(num_channels, sigma, dtype=dtype)

        # Create ImageDerivative module
        self.derivative = ImageDerivative(num_channels, dtype=dtype)

        # Whether to use torch.gather to get the gradient and hessian at the max location
        self.use_gather = use_gather

        # Whether to clamp the output to be within the heatmap
        self.clamp_output = clamp_output

    def forward(self, heatmap:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Finds the location and value of the maximum value in each heatmap channel using the DARK method:
            1. Smooth heatmap if sigma is not None
            2. Find the max value and location in each heatmap channel 
            2. Compute log of heatmap
            3. Compute first and second derivatives
            4. Get gradient and hessian at each max location
            5. Compute landmark w/ taylor series at max locations
            6. Return landmarks and max values
        Args:
            heatmap (torch.Tensor): heatmap of shape (batch_size, num_landmarks, height, width)
        Returns:
            landmarks (torch.Tensor): tensor of shape (batch_size, num_landmarks, 2) where the last dimension is (x, y)
            values (torch.Tensor): tensor of shape (batch_size, num_landmarks) where each value is the value of the max in the heatmap
        '''
        B, N, H, W = heatmap.shape

        # Get the max value and location in each heatmap channel
        max_values, max_indices = torch.max(heatmap.reshape(B, N, -1), dim=2)

        # Smooth heatmap if sigma is not None
        if hasattr(self, 'blur'):
            # Get max and min of each heatmap (want to rescale to original min and max)
            min_h = torch.min(heatmap.flatten(start_dim=-2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)

            # Blur heatmap
            heatmap = self.blur(heatmap)

            # Normalize heatmap to be between 0 and 1
            min_new_h = torch.min(heatmap.flatten(start_dim=-2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
            max_new_h = torch.max(heatmap.flatten(start_dim=-2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
            heatmap = (heatmap - min_new_h) / (max_new_h - min_new_h)

            # Rescale heatmap to be between min and max of original heatmap
            heatmap = heatmap * (max_values.unsqueeze(-1).unsqueeze(-1) - min_h) + min_h

        # Do log of heatmap
        heatmap = torch.clamp(heatmap, min=1e-8)
        heatmap = torch.log(heatmap)

        # Compute first and second derivatives
        image_der = self.derivative(heatmap) # B, 5N->(dx_i, dy_i, dxx_i, dyy_i, dxy_i), H, W
        image_der = image_der.view(B, N, 5, H, W)

        # Get gradient at each max location
        # -- max_indices: B, N where each value is the index of the max value in the FLATTENED heatmap
        # -- image_der: B, N, 5, H, W
        # -- grad_x: B, N where each value is the gradient in the x direction at the max location
        #    grad_x[b, n] = image_der[b, n, 0, max_indices[b, n] // W, max_indices[b, n] % W]
        if self.use_gather:
            grad_x = torch.gather(image_der[:, :, 0].flatten(start_dim = -2), dim=-1, index=max_indices.unsqueeze(-1)).squeeze(-1)
            grad_y = torch.gather(image_der[:, :, 1].flatten(start_dim = -2), dim=-1, index=max_indices.unsqueeze(-1)).squeeze(-1)
            hessian_xx = torch.gather(image_der[:, :, 2].flatten(start_dim = -2), dim=-1, index=max_indices.unsqueeze(-1)).squeeze(-1)
            hessian_yy = torch.gather(image_der[:, :, 3].flatten(start_dim = -2), dim=-1, index=max_indices.unsqueeze(-1)).squeeze(-1)
            hessian_xy = torch.gather(image_der[:, :, 4].flatten(start_dim = -2), dim=-1, index=max_indices.unsqueeze(-1)).squeeze(-1)

            grad = torch.stack((grad_x, grad_y), dim=-1)
            hessian = torch.stack((hessian_xx, hessian_yy, hessian_xy), dim=-1)
        # If not using gather (for exporting model), manually get the gradient and hessian at the max location
        else:
            grad = torch.zeros(B, N, 2, dtype=heatmap.dtype)
            grad = torch.zeros(B, N, 2,dtype=heatmap.dtype)
            hessian = torch.zeros(B, N, 3, dtype=heatmap.dtype)
            for b in range(B):
                for n in range(N):
                    grad[b, n, 0] = image_der[b, n, 0, max_indices[b, n] // W, max_indices[b, n] % W]
                    grad[b, n, 1] = image_der[b, n, 1, max_indices[b, n] // W, max_indices[b, n] % W]
                    hessian[b, n, 0] = image_der[b, n, 2, max_indices[b, n] // W, max_indices[b, n] % W]
                    hessian[b, n, 1] = image_der[b, n, 3, max_indices[b, n] // W, max_indices[b, n] % W]
                    hessian[b, n, 2] = image_der[b, n, 4, max_indices[b, n] // W, max_indices[b, n] % W]

        # Convert max_indices to x and y coordinates
        max_indices = max_indices.unsqueeze(2)
        max_indices = torch.cat((max_indices % W, max_indices // W), dim=2).type(heatmap.dtype)

        # Find hessians that are invertible
        det_hessian = hessian[:, :, 0] * hessian[:, :, 1] - hessian[:, :, 2] ** 2
        mask = torch.abs(det_hessian) > 1e-5

        # Also skip if the max point is within 2 pixels of the edge since the hessian estimation using a kernel of size 4
        num_pixel_dist = 2
        mask = mask & (max_indices[:, :, 0] >= num_pixel_dist) & (max_indices[:, :, 0] <= W - num_pixel_dist - 1) \
                    & (max_indices[:, :, 1] >= num_pixel_dist) & (max_indices[:, :, 1] <= H - num_pixel_dist - 1)

        # Compute landmark with taylor series at max locations
        #   -- First compute the inverse of the hessian
        inv_hessian = torch.stack((hessian[:, :, 1], -hessian[:, :, 2], 
                                   -hessian[:, :, 2], hessian[:, :, 0]), dim=-1).view(B, -1, 2, 2) / det_hessian.unsqueeze(-1).unsqueeze(-1)
        #  -- Compute the landmark using the taylor series: x = x - H^-1 * grad
        max_indices = max_indices - torch.einsum('bnxy, bny -> bnx', inv_hessian, grad) * mask.unsqueeze(-1).type(heatmap.dtype)

        # Clip max_indices to be within the heatmap
        if self.clamp_output:
            max_indices[..., 0] = torch.clamp(max_indices[..., 0], min=0, max=W - 1)
            max_indices[..., 1] = torch.clamp(max_indices[..., 1], min=0, max=H - 1)

        return max_indices, max_values