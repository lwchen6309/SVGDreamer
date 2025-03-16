import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import textwrap
import yaml
from svgdreamer.utils.composition_generator import generate_golden_spiral_image, generate_equal_lateral_triangle, \
    generate_diagonal_line, generate_l_shape_line, gaussian_filter

if __name__ == '__main__':
    
    # Create output directory
    output_dir = "edge_results"
    os.makedirs(output_dir, exist_ok=True)

    golden_spiral_image = generate_golden_spiral_image()
    triangle_image = generate_equal_lateral_triangle()
    diagonal_line_image = generate_diagonal_line()
    l_shape_image = generate_l_shape_line()
    composition_type_map = {
        "golden_spiral": golden_spiral_image,
        "pyramid": triangle_image,
        "diagonal": diagonal_line_image,
        "l_shape": l_shape_image
    }
    
    image_paths = glob.glob("logs/great_wall/sigma_25/SVGDreamer-*/sd*/all_particles.png")
    # image_paths = glob.glob("logs/great_wall/iter_2000/SVGDreamer-*/sd*/all_particles.png")

    # Set up subplots
    num_images = len(image_paths)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 6))
    if num_images == 1:
        axes = [axes]  # Ensure axes is iterable

    # Sobel filter kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    for ax, image_path in zip(axes, image_paths):
        # Load image
        image = Image.open(image_path).convert('RGB')
        images = transforms.ToTensor()(image).unsqueeze(0)  # Add batch dimension
        
        # Convert to grayscale
        images = images.mean(dim=1, keepdim=True)  # Average over the channels to get 1 channel
        
        # Apply Sobel filters
        edges_x = F.conv2d(images, sobel_x, padding=1, stride=1)  # Apply Sobel filter for x-direction
        edges_y = F.conv2d(images, sobel_y, padding=1, stride=1)  # Apply Sobel filter for y-direction
        
        # Compute magnitude of the gradient
        edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-6)  # Add small epsilon to avoid sqrt(0)
        max_vals = edges.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        edges = edges / max_vals  # Normalize the edges
        
        edges = edges.repeat(1, 3, 1, 1)  # Repeat the edges to match the number of channels in the input images
        edges = transforms.ToPILImage()(edges[0].cpu())
        
        # Plot edge image
        ax.imshow(edges, cmap='gray')
        ax.axis('off')
        
        # Extract composition type and sigma from overrides.yaml
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), ".hydra", "overrides.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                overrides = yaml.safe_load(f)
            composition_type = next((line.split("=")[-1] for line in overrides if "x.composition_loss.composition_type" in line), "golden_spiral").strip()
            sigma = next((line.split("=")[-1] for line in overrides if "x.composition_loss.sigma" in line), "50").strip()
            sigma = float(sigma)
            title_text = f"{composition_type}: Sigma: {sigma}"

            attention = torch.tensor(gaussian_filter(composition_type_map[composition_type], sigma=sigma))
            attention = attention.unsqueeze(0).unsqueeze(0).to(dtype=images.dtype, device=images.device)
            attention = F.interpolate(attention, size=images.shape[-2:], mode='bilinear', align_corners=False)
            attention = attention / attention.max()  # Normalize the attention map
            
            # Multiply the attention with the edges, and save it
            attention_weighted_edge = attention * transforms.ToTensor()(edges).to(attention.device)
            attention_weighted_edge = transforms.ToPILImage()(attention_weighted_edge.squeeze(0).cpu())
            attention_weighted_edge.save(os.path.join(output_dir, f"attn_w_edge_{composition_type}.png"))
            edges.save(os.path.join(output_dir, f"edges_{composition_type}.png"))
            
            # Save attention map as an image
            attention_map = transforms.ToPILImage()(attention.squeeze(0).cpu())
            attention_map.save(os.path.join(output_dir, f"attn_{composition_type}.png"))
        else:
            title_text = "Unknown"
        ax.set_title("\n".join(textwrap.wrap(title_text, width=20)), fontsize=20)

    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "edges_comparison.png"))
