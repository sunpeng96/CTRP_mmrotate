# -*- coding: utf-8 -*-
# @Time    : 02/20/2024
# @Author  : Peng Sun
# @FileName: img_random_mask_v2.py 遇到图像中实例较少(只有 1个)的情况, 不需要掩膜
# @Software: PyCharm

import cv2
import torch
import numpy as np

from mmrotate.core.bbox import obb2poly

def random_masking(img, img_metas,
                   gt_bboxes_list,
                   gt_labels_list,
                   mask_ratio):
    """
    Args:
        num_mask: Number of masked ground truth

    Returns:
        img_mask: (Tensor): of shape (N, C, H, W) encoding input images.
    """

    assert len(gt_bboxes_list) == len(gt_labels_list)

    new_img_list = []
    new_gt_bboxes_list = []
    new_gt_labels_list = []

    for i, (img_batch, gt_bboxes, gt_labels, img_meta) in enumerate(zip(img, gt_bboxes_list, gt_labels_list, img_metas)):

        assert len(gt_bboxes) == len(gt_labels) and len(gt_bboxes)>0

        img_shape = img_meta['img_shape'][:2]  # The size of the Input Image, (h, w)
        img_device = img_batch.device

        num_gt = len(gt_bboxes)
        if num_gt == 1:  # 当图像中只有一个 GT时, 不对图像作掩膜处理;
            new_img_batch = img_batch
        else:
            num_mask = int(num_gt * 0.25) if (num_gt >= 1 and num_gt < 20) else int(num_gt * mask_ratio)
            # 当 GT的数量在[2, 20)的区间的时候，取 25%的 GT做 mask处理;
            # 当 GT>=20时, 取小于 50%的一个比例的 GT做mask, 这个比例作为一个可以调整的超参数;
            noise = torch.rand(num_gt, device=img_device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=0)  # ascend: small is masked

            # ids_keep = ids_shuffle[num_mask:]  # keep the first subset for keep ids
            ids_mask = ids_shuffle[:num_mask]  # keep the remaining subset for mask ids

            # Extract the keep bbox and label
            # gt_bboxes_keep = gt_bboxes[ids_keep, :]  # (Tensor) shape (num_gt - num_maks, 5)
            # gt_labels_keep = gt_labels[ids_keep]  # (Tensor) shape (num_gt - num_maks, 1)

            # Extract the maks bbox and label
            gt_bboxes_mask = gt_bboxes[ids_mask, :]  # (Tensor) shape (num_maks, 5)
            gt_labels_mask = gt_labels[ids_mask]  # (Tensor) shape (num_maks, 1)

            gt_labels[ids_mask] = torch.tensor([15], device=gt_labels.device)

            mask = np.zeros(img_shape, dtype=np.uint8)  # 构造一个与输入图像尺寸一致的掩膜图像
            # 将被 mask的真值的表达形式转换为多边形形式，并构建根据真值构建响应的掩膜
            polys_mask = obb2poly(gt_bboxes_mask, version='le90').tolist()
            for poly in polys_mask:
                poly_split = [poly[i:i + 2] for i in range(0, len(poly), 2)]
                poly_np = np.array(poly_split, np.int32)
                cv2.fillPoly(mask, [poly_np], (255, 255, 255))  # 在 gt_mask位置填充掩膜
            mask = torch.tensor(mask, device=img_device).reshape(1, img_shape[0], img_shape[1]).repeat(3, 1, 1).bool()
            # 将掩膜和图像融合, 使用的是基于 pytorch的方法
            # img_mask_bs = cv2.add(img_bs, np.zeros(np.shape(img_bs)), mask=mask)
            # img_mask_bs = cv2.bitwise_and(img_bs, img_bs, mask=mask)
            new_img_batch = torch.masked_fill(input=img_batch, mask=mask, value=0)

        # 测试用例
        # mask = torch.zeros((1024, 1024), device=img.device).reshape(1, 1024, 1024).repeat(3, 1, 1).bool()
        # img_mask_bs = torch.masked_fill(input=img_bs, mask=~mask, value=0)

        new_img_list.append(new_img_batch.unsqueeze(dim=0))  # List (Tensor)

        new_gt_bboxes_list.append(gt_bboxes)  # List[Tensor]: [batch_size, (num_gt, 5)]
        new_gt_labels_list.append(gt_labels)  # List[Tensor]: [batch_size, (num_gt, 1)]

    new_img = torch.cat(new_img_list, dim=0)  # Tensor: shape (2, 1024, 1024, 3)

    return new_img, new_gt_bboxes_list, new_gt_labels_list