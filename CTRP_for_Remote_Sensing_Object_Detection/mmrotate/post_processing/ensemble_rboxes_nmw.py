# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

"""
Method described in:
CAD: Scale Invariant Framework for Real-Time Object Detection
http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w14/Zhou_CAD_Scale_Invariant_ICCV_2017_paper.pdf
"""

# import warnings
# import numpy as np
# from numba import jit

import torch
import torch.nn.functional as F
from mmcv.ops import nms_rotated

from mmrotate.structures.bbox.bbox_overlaps import rbbox_overlaps
from mmrotate.structures.bbox.gauss_represent import (gauss2rbox, rbox2gauss)


def fast_synth_box(boxes, scores, labels, synth_thr, main_idx, ious, method=2, alpha=2.0, beta=1.0,
                   use_weight_scores=False):
    """
    :param boxes:   torch.Tensor, boxes, shape [N, 5]
    :param scores:  torch.Tensor, scores for boxes [N,]
    :param labels:  torch.Tensor, labels for boxes [N,]
    :param synth_thr:  Float, threshold for synth
    :param main_idx: main boxes indexes, use nms to get it. [m]
    :param potential_idx: the boxes are not in main boxes. [p]
    :param ious: the iou between main boxes and all boxes. [m, N]
    :param method: 1 for directly weight boxes, 2 for gaussian weight boxes.
    :param alpha: the exponential term of iou.
    :param beta: the exponential term of score.
    :param use_weight_scores: the flag for weight scores

    :return: boxes: boxes (N, 5).
    :return: scores: confidence scores (N, )
    """
    max_iou, idx = torch.max(ious, dim=0)
    valid_pattern = max_iou > synth_thr

    valid_iou = ious[:, valid_pattern]
    num_main = main_idx.numel()
    num_all = valid_pattern.nonzero().numel()
    valid_mask = F.one_hot(idx[valid_pattern]).t().float()
    all_box = boxes[valid_pattern, :]
    all_score = scores[valid_pattern][None]
    all_iou = valid_iou
    all_weight = all_iou.pow(alpha) * all_score.pow(beta).expand([num_main, num_all]) * valid_mask
    weight_sum = all_weight.sum(dim=1)[..., None].expand(num_main, 5)
    all_weight = all_weight[..., None].expand([num_main, num_all, 5])
    if method == 2:
        box_ = rbox2gauss(all_box)[None].expand([num_main, num_all, 5])
    else:
        box_ = all_box[None].expand([num_main, num_all, 5])
    synth_box = torch.sum(all_weight * box_, dim=1) / weight_sum
    synth_scores = scores[main_idx].view(-1)
    synth_labels = labels[main_idx].view(-1)
    if method == 2:
        return gauss2rbox(synth_box), synth_scores, synth_labels
    else:
        return synth_box, synth_scores, synth_labels


def fast_rbox_synth_method(boxes, scores, labels, iou_thr=0.1, thresh=0.05, max_num=-1, check=False,
                           synth_thr=0.5, synth_method=2, alpha=2.0, beta=1.0,
                           use_weight_scores=False):
    """
    :param boxes:   torch.Tensor, boxes, shape [N, 5]
    :param scores:  torch.Tensor, scores for boxes [N,]
    :param labels:  torch.Tensor, labels for boxes [N,]
    :param iou_thr: IoU value for boxes to be a match
    :param thresh: Confidence threshold for boxes to keep

    :return: boxes: boxes (N, 5).
    :return: scores: confidence scores (N, )
    :return: labels: boxes labels (N, )
    """
    if boxes.shape[0] == 0:
        return boxes, scores, labels
    if check:
        valid_idx = scores >= thresh
        boxes = boxes[valid_idx]
        scores = scores[valid_idx]
        labels = labels[valid_idx]
    # t1 = time.time()
    max_coordinate = boxes[:, :2].max() + boxes[:, 2:4].max()
    offsets = labels.to(boxes) * (max_coordinate + 1)

    boxes_for_nms = boxes.clone()
    boxes_for_nms[:, :2] = boxes_for_nms[:, :2] + offsets[:, None]
    _, keep = nms_rotated(boxes_for_nms.clone(), scores, iou_thr)
    if max_num > 0:
        keep = keep[:max_num]
    main_bboxes_for_iou = boxes_for_nms[keep]
    # print(f'NMS Time: {(time.time() - t1) * 1000: .2f} ms')
    # t = time.time()
    # 相同的框不一定会正确计算IOU，出现了同一个框之间的IOU为0.333的情况。因此需要在计算完成后强制赋值。
    # mmcv 尚未修复该BUG,只能强制修改赋值
    iou_table = rbbox_overlaps(main_bboxes_for_iou.contiguous(), boxes_for_nms.contiguous(), 'iou', False)
    mask = F.one_hot(keep, num_classes=scores.numel()).bool()
    iou_table[mask] = 1.0
    # iou_table =
    # print(f'IoU Time: {(time.time() - t) * 1000: .2f} ms')
    synth_boxes, synth_scores, synth_labels = fast_synth_box(boxes,
                                                             scores,
                                                             labels,
                                                             synth_thr,
                                                             keep,
                                                             iou_table,
                                                             method=synth_method,
                                                             alpha=alpha,
                                                             beta=beta,
                                                             use_weight_scores=use_weight_scores)
    # print(f'Synth Time: {(time.time() - t) * 1000: .2f} ms')
    return synth_boxes, synth_scores, synth_labels