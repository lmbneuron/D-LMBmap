import torch
from torch.nn.functional import affine_grid, grid_sample
import numpy as np


__all__ = ["to2delta", "delta2to"]


def to2delta(deform_space, image_shape):
    """
    in pytorch, deform space is point to point
    while in elastix it's the offset of points
    """
    base_transform = generate_base_deform_space(image_shape,
                                                deform_space.device)
    deform_space = deform_space - base_transform
    return deform_space


def delta2to(delta_deform_space, image_shape):
    """
    in pytorch, deform space is point to point
    while in elastix it's the offset of points
    """
    base_transform = generate_base_deform_space(image_shape,
                                                delta_deform_space.device)
    delta_deform_space = delta_deform_space + base_transform
    return delta_deform_space


def generate_base_deform_space(shape, device):
    """
    generate a base deform space to calculate points' offset
    """
    a = torch.tensor([[[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]]], dtype=torch.float32, device=device)
    a = a.repeat(shape[0], 1, 1)
    affine = affine_grid(a, shape)
    return affine


def flowinverse(flow, is_delta=True):
    """
    calculate a reverse deform space, flow shall be offset instead of coordinate
    """
    if not is_delta:
        flow = to2delta(flow, (flow.shape[1], flow.shape[2], flow.shape[3]))
    flow0 = torch.unsqueeze(flow[:, :, :, :, 0], 1)
    flow1 = torch.unsqueeze(flow[:, :, :, :, 1], 1)
    flow2 = torch.unsqueeze(flow[:, :, :, :, 2], 1)

    flow0 = grid_sample(flow0, flow, align_corners=True)
    flow1 = grid_sample(flow1, flow, align_corners=True)
    flow2 = grid_sample(flow2, flow, align_corners=True)

    flow_inverse = torch.cat((flow0, flow1, flow2), 1)
    flow_inverse = flow_inverse.permute(0, 2, 3, 4, 1)
    flow_inverse *= -1
    if not is_delta:
        flow_inverse = delta2to(flow_inverse, (flow.shape[1], flow.shape[2], flow.shape[3]))
    return flow_inverse


def grid_sample_nearest(data, deform_space):
    assert deform_space.shape[0] == 1, "only can be used when batch size is 1"

    result_shape = data.shape[2:]
    deform_space_npy = deform_space.cpu().numpy().squeeze(0)
    to_x = (deform_space_npy[:, :, :, 2] + 1) * (result_shape[0] - 1) / 2
    to_y = (deform_space_npy[:, :, :, 1] + 1) * (result_shape[1] - 1) / 2
    to_z = (deform_space_npy[:, :, :, 0] + 1) * (result_shape[2] - 1) / 2
    raw_img = data.cpu().numpy().squeeze()
    result_img = np.zeros_like(raw_img)
    for i in range(np.shape(deform_space_npy)[0]):
        for j in range(np.shape(deform_space_npy)[1]):
            for k in range(np.shape(deform_space_npy)[2]):
                x = int(round(to_x[i, j, k]))
                y = int(round(to_y[i, j, k]))
                z = int(round(to_z[i, j, k]))
                result_img[i, j, k] = raw_img[x, y, z]
    return torch.tensor(result_img[np.newaxis, np.newaxis, ...], dtype=data.dtype,
                                   device=data.device)
