import torch
import torch.nn.functional as F
import numpy as np


# Sobel filter kernels for edge detection (x and y directions)
sobel_kernel_x = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], requires_grad=False)
sobel_kernel_y = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]], requires_grad=False)


def gaussian_kernel(kernel_size: int, sigma: float):
    """
    Generate a 2D Gaussian kernel.
    
    Arguments:
    - kernel_size: Size of the Gaussian kernel (odd number).
    - sigma: Standard deviation for Gaussian distribution.
    
    Returns:
    - kernel: A 2D Gaussian kernel tensor.
    """
    # Create a 1D Gaussian kernel
    kernel_1d = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    kernel_1d = np.exp(-(kernel_1d**2) / (2 * sigma**2))
    
    # Normalize the kernel
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create a 2D Gaussian kernel by outer product of 1D kernel
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    # Normalize the kernel to ensure sum is 1
    kernel_2d = kernel_2d / kernel_2d.sum()

    # Convert to torch tensor and return
    return torch.tensor(kernel_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def apply_gaussian_blur(input_tensor, kernel_size=5, sigma=1.0):
    """
    Apply a Gaussian blur to a tensor using a manually created Gaussian kernel.
    
    Arguments:
    - input_tensor: The input tensor to blur (shape: [batch_size, channels, height, width]).
    - kernel_size: Size of the Gaussian kernel (odd number).
    - sigma: Standard deviation for Gaussian distribution.
    
    Returns:
    - blurred_tensor: The blurred tensor.
    """
    # Generate the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    
    # Ensure the kernel is in the same device as the input tensor
    kernel = kernel.to(input_tensor.device)
    
    # Apply the Gaussian kernel using 2D convolution (input shape: [batch_size, channels, height, width])
    blurred_tensor = F.conv2d(input_tensor, kernel, padding=kernel_size//2, groups=input_tensor.shape[1])
    
    return blurred_tensor

def apply_sobel_to_masks(pred_masks, kernel_size=None, sigma=1.0):
    """
    Applies Gaussian blur to each mask before performing Sobel edge detection, with batch processing.

    Arguments:
    - pred_masks: The segmentation masks to apply Sobel edge detection on. Shape: [batch_size, num_classes, height, width]
    - kernel_size: Size of the Gaussian kernel (optional).
    - sigma: Standard deviation of the Gaussian distribution (optional).

    Returns:
    - sobel_masks: Sobel-edged masks after blurring.
    """
    batch_size, num_classes, height, width = pred_masks.shape
    device = pred_masks.device
    sobel_masks = []

    # Define Sobel kernels for separable convolution (x and y directions)
    sobel_kernel_x = torch.tensor([[-1.0, 0.0, 1.0]], dtype=pred_masks.dtype, requires_grad=False).to(device)  # 1D horizontal kernel
    sobel_kernel_y = torch.tensor([[-1.0], [0.0], [1.0]], dtype=pred_masks.dtype, requires_grad=False).to(device)  # 1D vertical kernel

    # Reshape the Sobel kernels to have the required dimensions for conv2d
    sobel_kernel_x = sobel_kernel_x.view(1, 1, 1, 3)  # Shape: [out_channels, in_channels, height, width] -> [1, 1, 1, 3]
    sobel_kernel_y = sobel_kernel_y.view(1, 1, 3, 1)  # Shape: [out_channels, in_channels, height, width] -> [1, 1, 3, 1]
    
    for class_idx in range(num_classes):
        # Extract the mask for the current class (batch_size, height, width)
        class_mask = pred_masks[:, class_idx, :, :].unsqueeze(1)  # Shape: [batch_size, 1, height, width]

        if kernel_size is not None:
            # Apply Gaussian blur before Sobel edge detection (if kernel_size is provided)
            class_mask = apply_gaussian_blur(class_mask, kernel_size=kernel_size, sigma=sigma)

        # Apply separable Sobel filters (horizontal and vertical convolutions)
        sobel_x = F.conv2d(class_mask, sobel_kernel_x, padding=(0, 1), stride=1)  # Apply Sobel filter in x direction
        sobel_y = F.conv2d(class_mask, sobel_kernel_y, padding=(1, 0), stride=1)  # Apply Sobel filter in y direction

        # Compute the magnitude of the gradient (sqrt(gx^2 + gy^2))
        sobel_magnitude = torch.sqrt(sobel_x**2 + sobel_y**2 + 1e-6)  # Add small epsilon to avoid sqrt(0)

        # Append the Sobel edge result for the current class
        sobel_masks.append(sobel_magnitude.squeeze(1))  # Remove extra channel dimension

    # Stack the results for all classes into a [batch_size, num_classes, height, width] tensor
    return torch.stack(sobel_masks, dim=1)  # Shape: [batch_size, num_classes, height, width]
