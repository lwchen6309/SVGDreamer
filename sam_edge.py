import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from transformers import SamModel, SamProcessor
import numpy as np
import torchvision.transforms as transforms
from svgdreamer.utils.sobel_edge import apply_sobel_to_masks

# Function to apply Sobel edge detection
def apply_sobel(image):
    return image.filter(ImageFilter.FIND_EDGES)

# Convert PIL image to tensor
def pil_to_tensor(pil_img):
    return transforms.ToTensor()(pil_img).unsqueeze(0)  # Convert to [1, C, H, W]

def infer_sam_edge(images, model, processor, device, input_points = None, kernel_size=None, sigma=5.0):    
    # Process the image and input points
    inputs = processor(images=images, input_points=input_points, return_tensors="pt").to(device)
    
    # Get segmentation map using SAM model
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the mask (resize it to the original image size)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )

    # Get the segmentation mask by applying argmax across the class dimension
    segmentation_map = masks[0].cpu()  # Shape: [1, C, H, W]
    sobel_masks = apply_sobel_to_masks(segmentation_map.type(torch.float), kernel_size=kernel_size, sigma=sigma)
    return sobel_masks


if __name__ == "__main__":
    # Directory containing images
    image_dir = "scene_examples"

    # Get all image file names from the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Number of images to process (limit to first 10)
    n = min(10, len(image_files))  # Limit to first 10 images

    # Load pre-trained SAM model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/sam-vit-large"
    model = SamModel.from_pretrained(model_name).to(device)
    processor = SamProcessor.from_pretrained(model_name)
    
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
        
        sobel_masks = infer_sam_edge(img, model, processor, device, input_points = None)

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
        axes[2, i].imshow(sobel_masks[0].mean(dim=0), cmap=cmap)  # Use colormap for segmentation
        axes[2, i].axis('off')  # Turn off axis
        axes[2, i].set_title(f"Sobel of SegMap {i+1}")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot to a file (e.g., 'segmentation_output.png')
    plt.savefig('segmentation_output.png', dpi=300)  # Adjust the DPI for resolution (300 is high resolution)
    plt.close()  # Close the plot to free up memory

