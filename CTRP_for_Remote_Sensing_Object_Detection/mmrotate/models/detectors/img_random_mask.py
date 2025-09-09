# -*- coding: utf-8 -*-
# @Time    : 01/10/2024
# @Author  : Peng Sun
# @FileName: img_random_mask.py
# @Software: PyCharm
# @Describe:
#   掩膜颜色为白色
#   遇到图像中实例较少(只有 1个)的情况下, 复制实例制作掩膜, 输出是保留的 GT和被掩膜的 GT.

import cv2
import math
import mmcv
import torch
import numpy as np
from mmcv.image import tensor2imgs

from mmrotate.core.bbox import obb2poly, obb2hbb

def random_mask(img, img_metas, mask_ratio,
                   gt_bboxes_list, gt_labels_list):
    """
    Args:
        img (Tensor): shape (N, C, H, W).

        num_mask: Proportion of masked ground truth.

        gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
            shape (num_gts, 5) in [cx, cy, w, h, a] format.

        gt_labels (list[Tensor]): class indices corresponding to each box.

        img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

    Returns:
        img_mask: (Tensor): of shape (N, C, H, W) encoding input images.
    """

    assert len(gt_bboxes_list) == len(gt_labels_list)
    img_device = img.device

    mask_img_list = []
    mask_bboxes_list = []
    mask_labels_list = []
    new_gt_bboxes_list = []
    new_gt_labels_list = []

    # tensor2imgs: mmcv中的模块, 作用是将 Tensor转换为 Numpy结构.
    #   其中的重要操作是将图像根据 config中的 mean和 std归一化.
    img = tensor2imgs(img, **img_metas[0]['img_norm_cfg'])  # 将 Tensor的图像转换为 numpy
    for i, (img_batch, gt_bboxes, gt_labels, img_meta) in enumerate(zip(img, gt_bboxes_list, gt_labels_list, img_metas)):

        assert len(gt_bboxes) == len(gt_labels)

        num_gt = len(gt_bboxes)
        img_shape = img_meta['img_shape']  # The size of the Input Image, (h, w, c)

        if num_gt == 1:  # 当只有一个GT的时候, 计算这个GT的最小外接矩形并放大, 之后将这部分区域在图像中平移一定的距离, 再制造一个GT;
            init_area, shift_area, init_bbox, shift_bbox, scale_ratio = instance_repeat(gt_bboxes, img_shape)
            # 解码初始区域和移动区域的坐标, Notes: x和 y表示的是左下角点坐标;
            init_x0, init_y0, init_x1, init_y1 = init_area
            shift_x0, shift_y0, shift_x1, shift_y1 = shift_area

            roi = img_batch[init_y0:init_y1, init_x0:init_x1, :]
            img_batch[shift_y0:shift_y1, shift_x0:shift_x1, :] = roi[::scale_ratio, ::scale_ratio, :]

            gt_bboxes_mask = shift_bbox
            gt_labels_mask = gt_labels

            gt_bboxes_keep = gt_bboxes
            gt_labels_keep = gt_labels

            # gt_bboxes_mask = gt_bboxes
            # gt_labels_mask = gt_labels
            #
            # gt_bboxes_keep = gt_bboxes
            # gt_labels_keep = gt_labels

            # gt_bboxes = torch.cat([init_bbox, shift_bbox], dim=0)
            # gt_labels = torch.cat([gt_labels, torch.tensor([15], device=gt_labels.device)])

        else:
            num_mask = int(num_gt * 0.5) if (num_gt > 1 and num_gt <= 20) else int(num_gt * mask_ratio)
            # 当 GT的数量在[2, 20)的区间的时候，取 25%的 GT做 mask处理;
            # 当 GT>=20时, 取小于 50%的一个比例的 GT做mask, 这个比例作为一个可以调整的超参数;

            noise = torch.rand(num_gt, device=img_device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=0)  # ascend: small is masked

            ids_keep = ids_shuffle[num_mask:]  # keep the first subset for keep ids
            ids_mask = ids_shuffle[:num_mask]  # keep the remaining subset for mask ids

            # Extract the keep bbox and label
            gt_bboxes_keep = gt_bboxes[ids_keep, :]  # (Tensor) shape (num_gt - num_maks, 5)
            gt_labels_keep = gt_labels[ids_keep]  # (Tensor) shape (num_gt - num_maks, 1)

            # Extract the maks bbox and label
            gt_bboxes_mask = gt_bboxes[ids_mask, :]  # (Tensor) shape (num_maks, 5)
            gt_labels_mask = gt_labels[ids_mask]  # (Tensor) shape (num_maks, 1)

            # gt_labels[ids_mask] = torch.tensor([15], device=gt_labels.device)

        # 将被 mask的真值的表达形式转换为多边形形式，并构建根据真值构建响应的掩膜;
        alpha = 1.0
        color = (255, 255, 255)

        if img_batch.shape!=img_shape:
            img_shape = img_batch.shape

        mask = np.zeros(img_shape, dtype=np.uint8)  # 构造一个与输入图像尺寸一致的掩膜图像;
        polys_mask = obb2poly(gt_bboxes_mask, version='le90').tolist()
        for poly in polys_mask:
            poly_split = [poly[i:i + 2] for i in range(0, len(poly), 2)]
            poly_np = np.array(poly_split, np.int32)
            cv2.fillPoly(mask, [poly_np], color)  # 在 gt_mask位置填充掩膜
        # 两张图片加权的方法, 需要将图片转换为 numpy格式

        new_img_batch_np = cv2.addWeighted(img_batch, 1.0, mask, alpha, 1)
        # 掩膜方法
        # mask = torch.tensor(mask, device=img_device).reshape(1, img_shape[0], img_shape[1]).repeat(3, 1, 1).bool()
        # new_img_batch = torch.masked_fill(input=img_batch, mask=mask, value=255)

        mean = np.array(img_metas[0]['img_norm_cfg']['mean'], dtype=np.float32)
        std = np.array(img_metas[0]['img_norm_cfg']['std'], dtype=np.float32)
        to_rgb = img_metas[0]['img_norm_cfg']['to_rgb']

        new_img_batch_np = mmcv.imnormalize(new_img_batch_np, mean, std, to_rgb).transpose(2, 0, 1)
        new_img_batch = torch.tensor(new_img_batch_np, device=img_device)

        # 测试用例
        # mask = torch.zeros((1024, 1024), device=img.device).reshape(1, 1024, 1024).repeat(3, 1, 1).bool()
        # img_mask_bs = torch.masked_fill(input=img_bs, mask=~mask, value=0)

        mask_img_list.append(new_img_batch)  # List (Tensor)
        new_gt_bboxes_list.append(gt_bboxes_keep)  # List[Tensor]: [batch_size, (num_gt, 5)]
        new_gt_labels_list.append(gt_labels_keep)  # List[Tensor]: [batch_size, (num_gt, 1)]
        mask_bboxes_list.append(gt_bboxes_mask)  # List[Tensor]: [batch_size, (num_mask, 5)]
        mask_labels_list.append(gt_labels_mask)  # List[Tensor]: [batch_size, (num_mask, 1)]

    mask_img = torch.stack(mask_img_list, dim=0)  # Tensor: shape (2, 1024, 1024, 3)

    rand_mask_results = {'mask_image': mask_img,
                         'mask_bboxes': mask_bboxes_list,
                         'mask_labels': mask_labels_list}

    return rand_mask_results, new_gt_bboxes_list, new_gt_labels_list

def instance_repeat(gt_bboxes, img_shape):

    init_bbox = gt_bboxes
    gt_hbb = obb2hbb(gt_bboxes, version='le90').squeeze(0).tolist()  # 将旋转框转换为水平框, 并压缩为 1维.
    gt_hbb = [int(gt_hbb[0]), int(gt_hbb[1]), int(gt_hbb[2]), int(gt_hbb[3])]

    gt_scale = math.sqrt(gt_hbb[2] * gt_hbb[3])  # 计算水平框的尺度, 用于判断调整区域的尺度因子 ratio.
    if img_shape[0]==1024: # For DOTA dataset
        eps=0
        if gt_scale > 0 and gt_scale <= 125:
            expand_ratio = 2; scale_ratio = 1; dist_factor = 1.5
        elif gt_scale > 256 and gt_scale <= 512:
            expand_ratio = 1; scale_ratio = 2; dist_factor = 2
        elif gt_scale > 512 and gt_scale <= 768:
            expand_ratio = 1; scale_ratio = 4; dist_factor = 2.5
        else:
            expand_ratio = 1; scale_ratio = 8; dist_factor = 3

    elif img_shape[1]==800: # For DIOR-R and HRSC2016 dataset
        eps = 0.5
        # print(gt_scale)
        if gt_scale > 0 and gt_scale <= 200:
            expand_ratio = 1; scale_ratio = 4; dist_factor = 4
            # print(11111)
        elif gt_scale > 200 and gt_scale <= 400:
            expand_ratio = 1; scale_ratio = 4; dist_factor = 4
            # print(22222)
        elif gt_scale > 400 and gt_scale <= 600:
            expand_ratio = 1; scale_ratio = 8; dist_factor = 6
            # print(33333)
        else:
            expand_ratio = 1; scale_ratio = 8; dist_factor = 6
            # print(44444)
    # 3. 计算初始区域的尺寸, 要注意转换为左下角点和长宽的形式, 这样便于后续计算
    gt_expand = adjust_ratio(gt_hbb, expand_ratio)  # in (x_ctr, y_ctr, w, h, angle) format.

    init_x0 = gt_expand[0] - gt_expand[2] / 2
    init_y0 = gt_expand[1] - gt_expand[3] / 2
    init_x1 = gt_expand[0] + gt_expand[2] / 2
    init_y1 = gt_expand[1] + gt_expand[3] / 2

    init_area = []
    for crood in [init_x0, init_y0, init_x1, init_y1]:
        crood = int(crood)
        if crood < 0:
            crood = 1
        if crood > img_shape[0]:
            crood = img_shape[0] - 1
        init_area.append(crood)

    init_xctr, init_yctr = (init_area[0] + init_area[2]) / 2, (init_area[1] + init_area[3]) / 2
    init_w, init_h = make_even(init_area[2] - init_area[0]), make_even(init_area[3] - init_area[1])

    init_area = [int(init_xctr - init_w / 2), int(init_yctr - init_h / 2),
                 int(init_xctr + init_w / 2), int(init_yctr + init_h / 2)]
    # Note: 为了便于计算, init_w, init_h 必须是偶数.

    # 4. 计算区域中心点移动的距离
    shift_dist = math.sqrt(math.pow(init_w, 2) + math.pow(init_h, 2)) / dist_factor

    # 为了保证移动后的区域不超界, 这里设定移动的方向为向图像中心移动
    delta_x, delta_y = int(img_shape[0]/2) - init_xctr, int(img_shape[0]/2) - init_yctr
    sqrt_xy = math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2)) + eps
    # 中心点移动的x, y轴距离的计算方法为, 首先计算初始区域的中心到图像中心的角度值，这样可以保证移动后的框在图中.
    shift_x, shift_y = int(shift_dist * (delta_x / sqrt_xy)), int(shift_dist * (delta_y / sqrt_xy))

    shift_xctr, shift_yctr = init_xctr + shift_x, init_yctr + shift_y
    shift_w, shift_h = init_w / scale_ratio, init_h / scale_ratio

    shift_area = [int(shift_xctr - shift_w / 2), int(shift_yctr - shift_h / 2),
                  int(shift_xctr + shift_w / 2), int(shift_yctr + shift_h / 2)]

    shift_bbox = [torch.tensor(shift_xctr, device=init_bbox.device).unsqueeze(0),
                  torch.tensor(shift_yctr, device=init_bbox.device).unsqueeze(0),
                  init_bbox[:, 2] / scale_ratio,
                  init_bbox[:, 3] / scale_ratio,
                  init_bbox[:, 4]]

    shift_bbox = torch.stack(shift_bbox, dim=1)

    return init_area, shift_area, init_bbox, shift_bbox, scale_ratio

def adjust_ratio(bboxes, ratio):
    '''Args:
        polys ([Tensor]): shape(n, 8)
        ratio: ratio of width and height
    Returns:
        polys ([Tensor])
    '''
    bboxes[2] *= ratio
    bboxes[3] *= ratio

    return bboxes

def make_even(num):
    if num % 8 == 0:
        return num
    else:
        return num - num % 8