# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch
from mmdet.structures.bbox import HorizontalBoxes, register_box_converter
from torch import Tensor

from .quadri_boxes import QuadriBoxes
from .rotated_boxes import RotatedBoxes, RRotatedBoxes
from .headpoint_boxes import HPBoxes, HPRotatedBoxes


@register_box_converter(HorizontalBoxes, RotatedBoxes)
def hbox2rbox(boxes: Tensor) -> Tensor:
    """Convert horizontal boxes to rotated boxes.

    Args:
        boxes (Tensor): horizontal box tensor with shape of (..., 4).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    wh = boxes[..., 2:] - boxes[..., :2]
    ctrs = (boxes[..., 2:] + boxes[..., :2]) / 2
    theta = boxes.new_zeros((*boxes.shape[:-1], 1))
    return torch.cat([ctrs, wh, theta], dim=-1)


@register_box_converter(HorizontalBoxes, QuadriBoxes)
def hbox2qbox(boxes: Tensor) -> Tensor:
    """Convert horizontal boxes to quadrilateral boxes.

    Args:
        boxes (Tensor): horizontal box tensor with shape of (..., 4).

    Returns:
        Tensor: Quadrilateral box tensor with shape of (..., 8).
    """
    x1, y1, x2, y2 = torch.split(boxes, 1, dim=-1)
    return torch.cat([x1, y1, x2, y1, x2, y2, x1, y2], dim=-1)


@register_box_converter(RotatedBoxes, HorizontalBoxes)
def rbox2hbox(boxes: Tensor) -> Tensor:
    """Convert rotated boxes to horizontal boxes.

    Args:
        boxes (Tensor): Rotated box tensor with shape of (..., 5).

    Returns:
        Tensor: Horizontal box tensor with shape of (..., 4).
    """
    ctrs, w, h, theta = torch.split(boxes, (2, 1, 1, 1), dim=-1)
    cos_value, sin_value = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * cos_value) + torch.abs(h / 2 * sin_value)
    y_bias = torch.abs(w / 2 * sin_value) + torch.abs(h / 2 * cos_value)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    return torch.cat([ctrs - bias, ctrs + bias], dim=-1)


@register_box_converter(RotatedBoxes, QuadriBoxes)
def rbox2qbox(boxes: Tensor) -> Tensor:
    """Convert rotated boxes to quadrilateral boxes.

    Args:
        boxes (Tensor): Rotated box tensor with shape of (..., 5).

    Returns:
        Tensor: Quadrilateral box tensor with shape of (..., 8).
    """
    ctr, w, h, theta = torch.split(boxes, (2, 1, 1, 1), dim=-1)
    cos_value, sin_value = torch.cos(theta), torch.sin(theta)
    vec1 = torch.cat([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
    vec2 = torch.cat([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return torch.cat([pt1, pt2, pt3, pt4], dim=-1)


@register_box_converter(QuadriBoxes, HorizontalBoxes)
def qbox2hbox(boxes: Tensor) -> Tensor:
    """Convert quadrilateral boxes to horizontal boxes.

    Args:
        boxes (Tensor): Quadrilateral box tensor with shape of (..., 8).

    Returns:
        Tensor: Horizontal box tensor with shape of (..., 4).
    """
    boxes = boxes.view(*boxes.shape[:-1], 4, 2)
    x1y1, _ = boxes.min(dim=-2)
    x2y2, _ = boxes.max(dim=-2)
    return torch.cat([x1y1, x2y2], dim=-1)


@register_box_converter(QuadriBoxes, RotatedBoxes)
def qbox2rbox(boxes: Tensor) -> Tensor:
    """Convert quadrilateral boxes to rotated boxes.

    Args:
        boxes (Tensor): Quadrilateral box tensor with shape of (..., 8).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    # TODO support tensor-based minAreaRect later
    original_shape = boxes.shape[:-1]
    points = boxes.cpu().numpy().reshape(-1, 4, 2)
    rboxes = []
    for pts in points:
        (x, y), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([x, y, w, h, angle / 180 * np.pi])
    rboxes = boxes.new_tensor(rboxes)
    return rboxes.view(*original_shape, 5)


@register_box_converter(QuadriBoxes, RRotatedBoxes)
def qbox2rrbox(boxes: Tensor) -> Tensor:
    """Convert quadrilateral boxes to rrotated boxes.

    Args:
        boxes (Tensor): Quadrilateral box tensor with shape of (..., 8).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    # TODO support tensor-based minAreaRect later
    original_shape = boxes.shape[:-1]
    points = boxes.cpu().numpy().reshape(-1, 4, 2)
    head_points = np.mean(points[:, :2, :], axis=1)
    ct_pionts = np.mean(points, axis=1)
    direction = (head_points - ct_pionts) / np.sqrt(np.sum(np.power((head_points - ct_pionts), 2.0), axis=1, keepdims=True))
    rboxes = []
    valid_range = [0.0, 360.0]
    for i, pts in enumerate(points):
        d = direction[i, :]
        (x, y), (w, h), angle = cv2.minAreaRect(pts)
        a_r = angle / 180.0 * np.pi
        a_r_1 = a_r + 0.5 * np.pi
        a_r_2 = a_r + np.pi
        a_r_3 = a_r + 1.5 * np.pi
        a_r_a = np.array([a_r, a_r_1, a_r_2, a_r_3], dtype=np.float32)
        r_y, r_x = np.sin(a_r_a), np.cos(a_r_a)
        m = r_x * d[0] + r_y * d[1]
        max_idx = np.argmax(m.reshape(-1))
        real_a = a_r_a[max_idx]
        if max_idx in [1, 3]:
            rboxes.append([x, y, h, w, real_a])
        else:
            rboxes.append([x, y, w, h, real_a])
    rboxes = boxes.new_tensor(rboxes)
    return rboxes.view(*original_shape, 5)


@register_box_converter(QuadriBoxes, HPBoxes)
def qbox2hpbox(boxes: Tensor) -> Tensor:
    """Convert quadrilateral boxes to rrotated boxes.

    Args:
        boxes (Tensor): Quadrilateral box tensor with shape of (..., 8).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    # TODO support tensor-based minAreaRect later
    # points = boxes.cpu().numpy().reshape(-1, 4, 2)

    boxes = boxes.view(*boxes.shape[:-1], 4, 2)
    head_points = boxes[:, :2, :].mean(1)
    x1y1, _ = boxes.min(dim=-2)
    x2y2, _ = boxes.max(dim=-2)
    return torch.cat([x1y1, x2y2, head_points], dim=-1)


@register_box_converter(QuadriBoxes, HPRotatedBoxes)
def qbox2hprbox(boxes: Tensor) -> Tensor:
    """Convert quadrilateral boxes to rotated boxes.

    Args:
        boxes (Tensor): Quadrilateral box tensor with shape of (..., 8).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    # TODO support tensor-based minAreaRect later
    original_shape = boxes.shape[:-1]
    points = boxes.cpu().numpy().reshape(-1, 4, 2)
    head_points = np.mean(points[:, :2, :], axis=1)
    head_points = torch.from_numpy(head_points.astype(np.float32))
    rboxes = []
    for pts in points:
        (x, y), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([x, y, w, h, angle / 180 * np.pi])
    rboxes = boxes.new_tensor(rboxes)
    return torch.cat([rboxes.view(*original_shape, 5), head_points], dim=1)


@register_box_converter(HPBoxes, HPRotatedBoxes)
def hpbox2hprbox(boxes: Tensor) -> Tensor:
    """Convert horizontal boxes to rotated boxes.

    Args:
        boxes (Tensor): horizontal box tensor with shape of (..., 4).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    wh = boxes[..., 2:4] - boxes[..., :2]
    ctrs = (boxes[..., 2:4] + boxes[..., :2]) / 2
    theta = boxes.new_zeros((*boxes.shape[:-1], 1))
    return torch.cat([ctrs, wh, theta, boxes[..., 4:6]], dim=-1)