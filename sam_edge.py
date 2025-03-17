import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from transformers import SamModel, SamProcessor
import numpy as np
import torchvision.transforms as transforms


# Sobel filter kernels for edge detection (x and y directions)
sobel_kernel_x = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], requires_grad=False)
sobel_kernel_y = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]], requires_grad=False)

# Convert PIL image to tensor
def pil_to_tensor(pil_img):
    return transforms.ToTensor()(pil_img).unsqueeze(0)  # Convert to [1, C, H, W]

# Function to apply Sobel edge detection
def apply_sobel(image):
    return image.filter(ImageFilter.FIND_EDGES)

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
    Applies Gaussian blur to each mask before performing Sobel edge detection.
    
    Arguments:
    - pred_masks: The segmentation masks to apply Sobel edge detection on.
    - kernel_size: Size of the Gaussian kernel.
    - sigma: Standard deviation of the Gaussian distribution.
    
    Returns:
    - sobel_masks: Sobel-edged masks after blurring.
    """
    batch_size, num_classes, height, width = pred_masks.shape
    sobel_masks = []
    
    for class_idx in range(num_classes):
        # Extract the mask for the current class (1 channel for each class)
        class_mask = pred_masks[0, class_idx, :, :].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, height, width]
        
        if kernel_size is not None:
        # Apply Gaussian blur before Sobel edge detection
            class_mask = apply_gaussian_blur(class_mask, kernel_size=kernel_size, sigma=sigma)
        
        # Sobel filter kernels for edge detection (x and y directions)
        sobel_kernel_x = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], requires_grad=False)
        sobel_kernel_y = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]], requires_grad=False)

        # Apply Sobel filters to detect edges in x and y directions
        sobel_x = F.conv2d(class_mask, sobel_kernel_x, padding=1)  # Apply Sobel filter in x direction
        sobel_y = F.conv2d(class_mask, sobel_kernel_y, padding=1)  # Apply Sobel filter in y direction

        # Compute the magnitude of the gradient (sqrt(gx^2 + gy^2))
        sobel_magnitude = torch.sqrt(sobel_x**2 + sobel_y**2)
        
        sobel_masks.append(sobel_magnitude.squeeze(0).squeeze(0))  # Remove extra dimensions, keep as tensor
    
    return torch.stack(sobel_masks)  # Stack the list of tensors to create a [num_classes, height, width] tensor


if __name__ == "__main__":
    # Directory containing images
    image_dir = "scene_examples"

    # Get all image file names from the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Number of images to process (limit to first 10)
    n = min(10, len(image_files))  # Limit to first 10 images

    # Load pre-trained SAM model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained("facebook/sam-vit-large").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
    
    # Calculate figure size based on number of images (adjust width per image and height per row)
    fig_width = 2 * n  # 4 inches per image
    fig_height = 7.5  # 10 inches for 4 rows (RGB, original, Sobel, and Segmentation)
    fig, axes = plt.subplots(3, n, figsize=(fig_width, fig_height))

    cmap = plt.get_cmap('viridis')  # Or use any other colormap like 'Set1', 'viridis', 'tab20', etc.

    # Loop over the images
    for i, image_file in enumerate(image_files[:n]):
        # Open the image
        img = Image.open(os.path.join(image_dir, image_file)).convert("RGB")
        
        # Convert to grayscale for Sobel edge detection
        grayscale_img = img.convert("L")
        
        # Apply Sobel edge detection
        sobel_img = apply_sobel(grayscale_img)
        
        # Get Sobel edge points (strongest edges)
        input_points = None  # Set to None to use default points
        
        # Process the image and input points
        inputs = processor(images=img, input_points=input_points, return_tensors="pt").to(device)
        
        # Get segmentation map using SAM model
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process the mask (resize it to the original image size)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        )

        # Get the segmentation mask by applying argmax across the class dimension
        segmentation_map = masks[0].cpu()  # Shape: [1, C, H, W]
        sobel_masks = apply_sobel_to_masks(segmentation_map.type(torch.float), kernel_size=21, sigma=5.0)
        # Plot RGB image in the first row
        axes[0, i].imshow(img)
        axes[0, i].axis('off')  # Turn off axis
        axes[0, i].set_title(f"Image {i+1}")
        
        # Plot original grayscale image in the second row
        # axes[1, i].imshow(grayscale_img, cmap='gray')
        # axes[1, i].axis('off')  # Turn off axis
        # axes[1, i].set_title(f"Greyscale {i+1}")
        
        # Plot Sobel edge image in the third row
        axes[1, i].imshow(sobel_img, cmap='gray')
        axes[1, i].axis('off')  # Turn off axis
        axes[1, i].set_title(f"Sobel {i+1}")
        
        # Plot segmentation map in the fourth row
        edge = pil_to_tensor(sobel_img)[0,0]
        axes[2, i].imshow(sobel_masks.mean(dim=0), cmap=cmap)  # Use colormap for segmentation
        axes[2, i].axis('off')  # Turn off axis
        axes[2, i].set_title(f"Sobel of SegMap {i+1}")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot to a file (e.g., 'segmentation_output.png')
    plt.savefig('segmentation_output.png', dpi=300)  # Adjust the DPI for resolution (300 is high resolution)
    plt.close()  # Close the plot to free up memory

