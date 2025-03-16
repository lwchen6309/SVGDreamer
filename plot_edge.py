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

def compute_edges(images):
    """Compute Sobel edges for the given images."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    edges_x = F.conv2d(images, sobel_x, padding=1, stride=1)
    edges_y = F.conv2d(images, sobel_y, padding=1, stride=1)
    edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-6)
    edges = edges / edges.max()  # Normalize
    edges = edges.repeat(1, 3, 1, 1)
    return transforms.ToPILImage()(edges[0].cpu())

def apply_attention(edges, composition_type, sigma, images):
    """Apply attention filter to edges based on composition type and sigma."""
    attention = torch.tensor(gaussian_filter(composition_type_map[composition_type], sigma=sigma))
    attention = attention.unsqueeze(0).unsqueeze(0).to(dtype=images.dtype, device=images.device)
    attention = F.interpolate(attention, size=images.shape[-2:], mode='bilinear', align_corners=False)
    attention = attention / attention.max()
    attention_weighted_edge = attention * transforms.ToTensor()(edges).to(attention.device)
    return transforms.ToPILImage()(attention_weighted_edge.squeeze(0).cpu()), transforms.ToPILImage()(attention.squeeze(0).cpu())

if __name__ == '__main__':
    output_dir = "edge_results"
    os.makedirs(output_dir, exist_ok=True)

    composition_type_map = {
        "golden_spiral": generate_golden_spiral_image(),
        "pyramid": generate_equal_lateral_triangle(),
        "diagonal": generate_diagonal_line(),
        "l_shape": generate_l_shape_line()
    }
    
    image_paths = glob.glob("logs/great_wall/sigma_50/SVGDreamer-*/sd*/all_particles.png")
    sorted_composition_types = ["golden_spiral", "pyramid", "diagonal", "l_shape"]
    
    experiments = []
    for image_path in image_paths:
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), ".hydra", "overrides.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                overrides = yaml.safe_load(f)
            composition_type = next((line.split("=")[-1] for line in overrides if "x.composition_loss.composition_type" in line), "golden_spiral").strip()
            sigma = float(next((line.split("=")[-1] for line in overrides if "x.composition_loss.sigma" in line), "50").strip())
            if composition_type in sorted_composition_types:
                experiments.append((composition_type, sigma, image_path))
    
    experiments.sort(key=lambda x: sorted_composition_types.index(x[0]))
    num_experiments = len(experiments)
    fig, axes = plt.subplots(num_experiments, 4, figsize=(20, 5 * num_experiments))
    
    titles = ["Images", "Edges", "Attention", "Attn Weighted Edge"]
    
    for i, (composition_type, sigma, image_path) in enumerate(experiments):
        image = Image.open(image_path).convert('RGB')
        images = transforms.ToTensor()(image).unsqueeze(0).mean(dim=1, keepdim=True)
        edges = compute_edges(images)
        attention_weighted_edge, attention_map = apply_attention(edges, composition_type, sigma, images)
        
        image.save(os.path.join(output_dir, f"all_particles_{composition_type}.png"))
        edges.save(os.path.join(output_dir, f"edges_{composition_type}.png"))
        attention_map.save(os.path.join(output_dir, f"attn_{composition_type}.png"))
        attention_weighted_edge.save(os.path.join(output_dir, f"attn_w_edge_{composition_type}.png"))
        
        axes[i, 0].imshow(image)
        axes[i, 1].imshow(edges, cmap='gray')
        axes[i, 2].imshow(attention_map, cmap='gray')
        axes[i, 3].imshow(attention_weighted_edge, cmap='gray')
        
        for j in range(4):
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=30)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "edges_comparison.png"))
