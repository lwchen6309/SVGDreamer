# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import torch
import torch.nn.functional as F
from svgdreamer.utils.sobel_edge import apply_sobel_to_masks


def channel_saturation_penalty_loss(x: torch.Tensor):
    assert x.shape[1] == 3
    r_channel = x[:, 0, :, :]
    g_channel = x[:, 1, :, :]
    b_channel = x[:, 2, :, :]
    channel_accumulate = torch.pow(r_channel, 2) + torch.pow(g_channel, 2) + torch.pow(b_channel, 2)
    return channel_accumulate.mean() / 3


def area(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def triangle_area(A, B, C):
    out = (C - A).flip([-1]) * (B - A)
    out = out[..., 1] - out[..., 0]
    return out


def compute_sine_theta(s1, s2):  # s1 and s2 aret two segments to be used
    # s1, s2 (2, 2)
    v1 = s1[1, :] - s1[0, :]
    v2 = s2[1, :] - s2[0, :]
    # print(v1, v2)
    sine_theta = (v1[0] * v2[1] - v1[1] * v2[0]) / (torch.norm(v1) * torch.norm(v2))
    return sine_theta


def compute_sine_theta_vectorized(s1, s2):  # s1 and s2 aret two segments to be used
    # s1 and s2 are tensors of shape (n, 2, 2) where n is the number of segment pairs
    # v1 and v2 are the vector differences between the start and end points of each segment
    v1 = s1[:, 1, :] - s1[:, 0, :]  # Shape (n, 2)
    v2 = s2[:, 1, :] - s2[:, 0, :]  # Shape (n, 2)
    
    # Compute the cross product (determinant) for each segment pair
    cross_product = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]  # Shape (n,)
    
    # Compute the norms of the vectors
    norm_v1 = torch.norm(v1, dim=1)  # Shape (n,)
    norm_v2 = torch.norm(v2, dim=1)  # Shape (n,)
    
    # Calculate the sine of the angle between the vectors
    sine_theta = cross_product / (norm_v1 * norm_v2)  # Shape (n,)
    
    return sine_theta


def xing_loss_fn_origin(x_list, scale=1e-3):  # x[npoints, 2]
    loss = 0.
    # print(f"points_len: {len(x_list)}")
    for x in x_list:
        # print(f"x: {x}")
        seg_loss = 0.
        N = x.size()[0]
        assert N % 3 == 0, f'The segment number ({N}) is not correct!'
        x = torch.cat([x, x[0, :].unsqueeze(0)], dim=0)  # (N+1,2)
        segments = torch.cat([x[:-1, :].unsqueeze(1), x[1:, :].unsqueeze(1)], dim=1)  # (N, start/end, 2)
        segment_num = int(N / 3)
        for i in range(segment_num):
            cs1 = segments[i * 3, :, :]  # start control segs
            cs2 = segments[i * 3 + 1, :, :]  # middle control segs
            cs3 = segments[i * 3 + 2, :, :]  # end control segs
            
            # print('the direction of the vectors:')
            # print(compute_sine_theta(cs1, cs2))
            direct = (compute_sine_theta(cs1, cs2) >= 0).float()
            opst = 1 - direct  # another direction
            sina = compute_sine_theta(cs1, cs3)  # the angle between cs1 and cs3
            seg_loss += direct * torch.relu(- sina) + opst * torch.relu(sina)
            # print(direct, opst, sina)
        seg_loss /= segment_num

        templ = seg_loss
        loss += templ * scale  # area_loss * scale

    return loss / (len(x_list))


def xing_loss_fn(x_list, scale=1e-3):  # x[npoints, 2]
    loss = 0.
    
    # Loop over the list of tensors in x_list
    for x in x_list:
        n = x.size(0)
        # assert n % 3 == 0, f'The segment number ({n}) is not correct!'
        
        # Prepare segments
        x = torch.cat([x, x[0, :].unsqueeze(0)], dim=0)  # (n+1,2)
        segments = torch.cat([x[:-1, :].unsqueeze(1), x[1:, :].unsqueeze(1)], dim=1)  # (n, start/end, 2)
        
        segment_num = n // 3
        
        # Precompute all directions and angles
        cs1 = segments[::3]  # start control segments
        cs2 = segments[1::3]  # middle control segments
        cs3 = segments[2::3]  # end control segments
        
        # Calculate the sine of the angles in a vectorized manner
        sina = compute_sine_theta_vectorized(cs1, cs3)
        direct = (compute_sine_theta_vectorized(cs1, cs2) >= 0).float()
        opst = 1 - direct  # another direction
        
        # Loss calculation without the inner loop
        seg_loss = torch.sum(direct * torch.relu(-sina) + opst * torch.relu(sina)) / segment_num
        
        # Add weighted loss
        loss += seg_loss * scale

    return loss / len(x_list)


def composition_loss_fn(images, composition_attention):
    """
    Computes the Sobel edge of images (applied to each channel separately) and calculates the dot product with
    the composition_attention along the spatial dimension using the mean.

    Parameters:
        images (torch.Tensor): The input image tensor for edge detection.
        composition_attention (torch.Tensor): The blurred golden spiral tensor.

    Returns:
        torch.Tensor: The result of the mean dot product between the Sobel edge and the spiral.
    """
    # Automatically determine the device from images
    device = images.device
    images = images.mean(dim=1, keepdim=True)  # Average over the channels to get 1 channel

    # 1. Compute Sobel edge for each channel separately
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=images.dtype, requires_grad=False).view(1, 1, 3, 3).to(device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=images.dtype, requires_grad=False).view(1, 1, 3, 3).to(device)

    # Apply Sobel filters to each channel of images (edge detection)
    edges_x = F.conv2d(images, sobel_x, padding=1, stride=1)  # Apply Sobel filter for x-direction
    edges_y = F.conv2d(images, sobel_y, padding=1, stride=1)  # Apply Sobel filter for y-direction
    
    # Combine the edges (magnitude of the gradient)
    edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-6)  # Add small epsilon to avoid sqrt(0)
    max_vals = edges.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    edges = edges / max_vals  # Normalize the edges

    # 2. Convert composition_attention to the same dtype and device as images
    composition_attention = composition_attention.to(dtype=images.dtype, device=device)
    composition_attention = F.interpolate(composition_attention, size=images.shape[-2:],
                                            mode='bilinear', align_corners=False)
    # Normalize the composition_attention
    max_vals = composition_attention.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    composition_attention = composition_attention / max_vals
    
    # 3. Compute the mean dot product along the spatial dimensions
    dot_product = -torch.mean(edges * composition_attention)
    
    # Return the computed dot product
    return dot_product


def sam_composition_loss_fn(images, composition_attention, model, processor, input_points=None, kernel_size=None, sigma=5.0):
    """
    Computes the Sobel edge of SAM prediction masks and calculates the dot product with
    the composition_attention along the spatial dimension using the mean.

    Parameters:
        images (torch.Tensor): The input image tensor for edge detection (used to calculate composition_attention).
        composition_attention (torch.Tensor): The blurred golden spiral tensor.
        model (SamModel): The SAM model used for segmentation.
        processor (SamProcessor): The processor used to pre-process the image for SAM.
        input_points (list): The list of input points for SAM segmentation.

    Returns:
        torch.Tensor: The result of the mean dot product between the Sobel edge and the spiral.
    """
    # Automatically determine the device from images
    device = images.device

    # 1. Process the image with SAM model to get segmentation mask
    inputs = processor(images=images, input_points=input_points, return_tensors="pt").to(device)

    # Get segmentation map using SAM model
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the mask (resize it to the original image size)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )

    # Get the segmentation mask by applying argmax across the class dimension
    segmentation_map = masks[0]  # Shape: [1, C, H, W]

    # 2. Compute Sobel edge for the segmentation map
    sobel_masks = apply_sobel_to_masks(segmentation_map.type(torch.float), kernel_size=kernel_size, sigma=sigma).mean(dim=1, keepdim=True)
    
    # 3. Normalize Sobel edges (as done with the image edges)
    max_vals = sobel_masks.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    sobel_masks = sobel_masks / max_vals  # Normalize the Sobel mask

    # 4. Convert composition_attention to the same dtype and device as images
    composition_attention = composition_attention.to(dtype=images.dtype, device=device)
    composition_attention = F.interpolate(composition_attention, size=images.shape[-2:], mode='bilinear', align_corners=False)

    # Normalize the composition_attention
    max_vals = composition_attention.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    composition_attention = composition_attention / max_vals

    # 5. Compute the mean dot product along the spatial dimensions
    dot_product = -torch.mean(sobel_masks * composition_attention)

    # Return the computed dot product as the loss
    return dot_product


if __name__ == '__main__':
    x_list = [torch.randn(6, 2) for _ in range(10)]
    torch.testing.assert_close(xing_loss_fn(x_list), xing_loss_fn_origin(x_list))