# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import torch
import torch.nn.functional as F
from svgdreamer.utils.composition_generator import generate_golden_spiral_image, generate_equal_lateral_triangle, \
    generate_diagonal_line, generate_l_shape_line, gaussian_filter


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


def compute_sine_theta(s1, s2):  # s1 and s2 aret two segments to be uswed
    # s1, s2 (2, 2)
    v1 = s1[1, :] - s1[0, :]
    v2 = s2[1, :] - s2[0, :]
    # print(v1, v2)
    sine_theta = (v1[0] * v2[1] - v1[1] * v2[0]) / (torch.norm(v1) * torch.norm(v2))
    return sine_theta


def xing_loss_fn(x_list, scale=1e-3):  # x[npoints, 2]
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


def composition_loss_fn(raster_imgs, blurred_golden_spiral):
    """
    Computes the Sobel edge of raster_imgs (applied to each channel separately) and calculates the dot product with
    the blurred_golden_spiral along the spatial dimension using the mean.

    Parameters:
        raster_imgs (torch.Tensor): The input image tensor for edge detection.
        blurred_golden_spiral (torch.Tensor): The blurred golden spiral tensor.

    Returns:
        torch.Tensor: The result of the mean dot product between the Sobel edge and the spiral.
    """
    # Automatically determine the device from raster_imgs
    device = raster_imgs.device
    raster_imgs = raster_imgs.mean(dim=1, keepdim=True)  # Average over the channels to get 1 channel

    # 1. Compute Sobel edge for each channel separately
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=raster_imgs.dtype, requires_grad=False).view(1, 1, 3, 3).to(device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=raster_imgs.dtype, requires_grad=False).view(1, 1, 3, 3).to(device)

    # Apply Sobel filters to each channel of raster_imgs (edge detection)
    edges_x = F.conv2d(raster_imgs, sobel_x, padding=1, stride=1)  # Apply Sobel filter for x-direction
    edges_y = F.conv2d(raster_imgs, sobel_y, padding=1, stride=1)  # Apply Sobel filter for y-direction

    # Combine the edges (magnitude of the gradient)
    edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-6)  # Add small epsilon to avoid sqrt(0)

    # 2. Convert blurred_golden_spiral to the same dtype and device as raster_imgs
    blurred_golden_spiral = blurred_golden_spiral.to(dtype=raster_imgs.dtype, device=device)
    blurred_golden_spiral = F.interpolate(blurred_golden_spiral, size=raster_imgs.shape[-2:],
                                            mode='bilinear', align_corners=False)

    # 3. Compute the mean dot product along the spatial dimensions
    dot_product = torch.mean(edges * blurred_golden_spiral)

    # Return the computed dot product
    return dot_product