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
from sam_edge import infer_sam_edge
from transformers import SamModel, SamProcessor


def compute_edges(images):
    """Compute Sobel edges for the given images.
    [B, C, H, W] -> [B, 1, H, W]"""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    edges_x = F.conv2d(images, sobel_x, padding=1, stride=1)
    edges_y = F.conv2d(images, sobel_y, padding=1, stride=1)
    edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-6)
    edges = edges / edges.max()  # Normalize
    return edges # [B, 1, H, W]

composition_type_map = {
    "golden_spiral": generate_golden_spiral_image,
    "pyramid": generate_equal_lateral_triangle,
    "diagonal": generate_diagonal_line,
    "l_shape": generate_l_shape_line
}

def apply_attention(edges, composition_type, sigma):
    """Apply attention filter to edges based on composition type and sigma.
    [B, 1, H, W] -> [B, 1, H, W], [B, 1, H, W]
    """
    target_comp = composition_type_map[composition_type]()

    attention = torch.tensor(gaussian_filter(target_comp, sigma=sigma))
    attention = attention.unsqueeze(0).unsqueeze(0).to(dtype=edges.dtype, device=edges.device)
    attention = F.interpolate(attention, size=edges.shape[-2:], mode='bilinear', align_corners=False)
    attention = attention / attention.max()
    attention_weighted_edge = attention * edges
    return attention_weighted_edge, attention
    

if __name__ == '__main__':
    output_dir = "edge_results"
    os.makedirs(output_dir, exist_ok=True)

    subdir = 'sam_edge_sigma_50'
    image_paths = glob.glob(f"logs/great_wall/{subdir}/SVGDreamer-*/sd*/all_particles.png")
    sorted_composition_types = ["golden_spiral", "pyramid", "diagonal", "l_shape"]
    save_fig = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/sam-vit-large"
    model = SamModel.from_pretrained(model_name).to(device)
    processor = SamProcessor.from_pretrained(model_name)

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
    fig, axes = plt.subplots(num_experiments, 6, figsize=(20, 5 * num_experiments))
    
    for i, (composition_type, sigma, image_path) in enumerate(experiments):
        image = Image.open(image_path).convert('RGB')
        images = transforms.ToTensor()(image).unsqueeze(0)
        
        grey_images = images.mean(dim=1, keepdim=True)
        edge = compute_edges(grey_images)
        attention_weighted_edge, attention_map = apply_attention(edge, composition_type, sigma)

        sam_edge = infer_sam_edge(images, model, processor, device, kernel_size=11).mean(dim=1, keepdim=True)
        sam_edge = sam_edge/sam_edge.max()
        attention_weighted_sam_edge, _ = apply_attention(sam_edge, composition_type, sigma)
        
        # List of images and their corresponding filenames
        images = [
            (image, "images"),
            (edge[0].cpu(), "edges"),
            (sam_edge[0].cpu(), "sam_edges"),
            (attention_map[0].cpu(), "attn"),
            (attention_weighted_edge[0].cpu(), "attn_w_edge"),
            (attention_weighted_sam_edge[0].cpu(), "attn_w_sam_edge"),
        ]

        # Save each image after converting to PIL if necessary
        if save_fig:
            for img, name in images:
                img = transforms.ToPILImage()(img) if isinstance(img, torch.Tensor) else img
                img.save(os.path.join(output_dir, f"{name}_{composition_type}.png"))

        # Display images on axes
        axes[i, 0].imshow(image)
        axes[i, 1].imshow(attention_map[0,0].cpu(), cmap='grey')
        axes[i, 2].imshow(edge[0,0].cpu(), cmap='grey')
        axes[i, 3].imshow(attention_weighted_edge[0,0].cpu(), cmap='grey')
        axes[i, 4].imshow(sam_edge[0,0].cpu()*100, cmap='viridis')
        axes[i, 5].imshow(attention_weighted_sam_edge[0,0].cpu(), cmap='viridis')
        titles = ["Images", "Attn", "Edges", "Attn WEdge", "SAM Edge", "Attn WSAMEdge"]
        for j, title in enumerate(titles):
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(title, fontsize=30)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "edges_comparison.png"))
