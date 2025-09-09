# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

# import numpy as np
from numba import jit
import torch

try:
    from ...structures.bbox.bbox_overlaps import rbbox_overlaps
    from ...structures.bbox import rbox2qbox
except:
    from mmrotate.structures.bbox.bbox_overlaps import rbbox_overlaps
    from mmrotate.structures.bbox import rbox2qbox

from .var import *


def soft_nms_float(dets, sc, Nt, thresh, method=3, sigma=0.5, iou_mode=False):
    """
    :param dets:   torch.Tensor, boxes, shape [N, 5]
    :param sc:     torch.Tensor, scores for boxes [N,]
    :param Nt:     Float, iou threshold
    :param sigma:  Float, parameters for gaussian soft nms
    :param thresh: Float, Confidence threshold
    :param method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS, 4 - DIoU NMS, 5 - CIoU NMS
    :return: index of boxes to keep
    """
    valid_idx = sc >= thresh
    dets = dets[valid_idx]
    sc = sc[valid_idx]
    dets_c = dets.contiguous()
    ious = rbbox_overlaps(dets_c, dets_c, 'iou', False)
    if method >= 4:
        x = dets_c[:, 0]
        y = dets_c[:, 1]
        n = x.numel()
        rou = (x.reshape(-1, 1).expand((n, n)) - x.reshape(1, -1).expand((n, n))).pow(2) + \
              (y.reshape(-1, 1).expand((n, n)) - y.reshape(1, -1).expand((n, n))).pow(2)
        poly_box = rbox2qbox(dets_c)
        xmin = torch.min(poly_box[:, 0::2], dim=1)[0]
        ymin = torch.min(poly_box[:, 1::2], dim=1)[0]
        xmax = torch.max(poly_box[:, 0::2], dim=1)[0]
        ymax = torch.max(poly_box[:, 1::2], dim=1)[0]
        x_range = torch.stack([xmin.reshape(-1, 1).expand((n, n)),
                               xmin.reshape(1, -1).expand((n, n)),
                               xmax.reshape(-1, 1).expand((n, n)),
                               xmax.reshape(1, -1).expand((n, n))], dim=-1)
        y_range = torch.stack([ymin.reshape(-1, 1).expand((n, n)),
                               ymin.reshape(1, -1).expand((n, n)),
                               ymax.reshape(-1, 1).expand((n, n)),
                               ymax.reshape(1, -1).expand((n, n))], dim=-1)
        x_min = torch.min(x_range, dim=2)[0]
        x_max = torch.max(x_range, dim=2)[0]
        y_min = torch.min(y_range, dim=2)[0]
        y_max = torch.max(y_range, dim=2)[0]
        c = (x_max - x_min).pow(2) + (y_max - y_min).pow(2)
        d = rou / c
        ious = ious - d

    N = dets.shape[0]
    indexes = torch.arange(N, device=dets.device)

    for i in range(N):
        if sc[i] < thresh:
            continue
        ovr = ious[i + 1:, i]
        keep_box = (ovr <= Nt)
        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            # weight = torch.ones(ovr.shape)
            weight = 1.0 - ovr
        elif method == 2:  # gaussian
            weight = torch.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = torch.zeros_like(ovr)
            weight[keep_box] = 1.0
        sc[i + 1:] = weight * sc[i + 1:]

    keep = indexes[sc > thresh]
    if iou_mode:
        return keep, ious[keep, :]
    else:
        return keep


def nms_method(boxes, scores, labels, method=3, iou_thr=0.5, sigma=0.5, thresh=0.001):
    """
    :param boxes:   torch.Tensor, boxes, shape [N, 5]
    :param scores:  torch.Tensor, scores for boxes [N,]
    :param labels:  torch.Tensor, labels for boxes [N,]
    :param method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS, 4 - DIoU NMS, 5 - CIoU NMS
    :param iou_thr: IoU value for boxes to be a match 
    :param sigma: Sigma value for SoftNMS
    :param thresh: Confidence threshold for boxes to keep (important for SoftNMS)

    :return: boxes: boxes (N, 5).
    :return: scores: confidence scores (N, )
    :return: labels: boxes labels (N, )
    """
    if SHOW_PARA:
        print(f'IoU thr: {iou_thr}\n')
    # Run NMS independently for each label
    unique_labels = torch.unique(labels)
    final_boxes = []
    final_scores = []
    final_labels = []
    for l in unique_labels:
        condition = (labels == l)
        boxes_by_label = boxes[condition]
        scores_by_label = scores[condition]
        # labels_by_label = torch.tensor([l] * len(boxes_by_label), dtype=torch.long, device=labels.device)
        labels_by_label = labels[condition]
        scores_descending, sort_idx = torch.sort(scores_by_label, descending=True)
        boxes_descending = boxes_by_label[sort_idx, :]

        keep = soft_nms_float(boxes_descending.clone(),
                              scores_descending.clone(),
                              iou_thr,
                              thresh,
                              sigma=sigma,
                              method=method)
        final_boxes.append(boxes_descending[keep])
        final_scores.append(scores_descending[keep])
        final_labels.append(labels_by_label[keep])
    final_boxes = torch.cat(final_boxes, dim=0)
    final_scores = torch.cat(final_scores)
    final_labels = torch.cat(final_labels)

    return final_boxes, final_scores, final_labels


def rotated_nms(boxes, scores, labels, iou_thr=0.5, score_thr=0.01, method=3):
    """
    Short call for standard NMS 
    
    :param boxes: 
    :param scores: 
    :param labels: 
    :param iou_thr:
    :return: 
    """
    return nms_method(boxes, scores, labels, method=method, iou_thr=iou_thr, thresh=score_thr)


def rotated_soft_nms(boxes, scores, labels, iou_thr=0.5, score_thr=0.05, sigma=0.5):
    """
    Short call for standard soft NMS

    :param boxes:
    :param scores:
    :param labels:
    :param iou_thr:
    :return:
    """
    return nms_method(boxes, scores, labels, method=2, iou_thr=iou_thr, thresh=score_thr, sigma=sigma)


def rotated_diou_nms(boxes, scores, labels, iou_thr=0.5, score_thr=0.01):
    """
    Short call for standard soft NMS

    :param boxes:
    :param scores:
    :param labels:
    :param iou_thr:
    :return:
    """
    return nms_method(boxes, scores, labels, method=4, iou_thr=iou_thr, thresh=score_thr)


def rotated_ciou_nms(boxes, scores, labels, angle_version, iou_thr=0.5, score_thr=0.01):
    """
    Short call for standard soft NMS

    :param boxes:
    :param scores:
    :param labels:
    :param iou_thr:
    :return:
    """
    return nms_method(boxes, scores, labels, method=5, iou_thr=iou_thr, thresh=score_thr, angle_version=angle_version)

# if __name__ == '__main__':
#     import os
#     import pickle
#     import torch
#     from mmrotate.core.visualization.image import imshow_det_rbboxes
#     dirs = '/workspace/mmrotate-0.3.2/work_dirs/CGR/test/rotated_retinanet_obb_r50_fpn_1x_dota_le90_cgr_test/image/'
#     pkl_file = 'P0016__1024__0___410.pkl'
#     # pkl_file = 'P1407__1024__0___0.pkl'
#     with open(os.path.join(dirs, pkl_file), 'rb') as f:
#         dets = pickle.load(f)
#         boxes, scores, labels = dets['boxes'], dets['scores'], dets['labels']
#         # boxes, scores, labels = rotated_nms(boxes, scores, labels, 0.1, 0.05)
#         # boxes, scores, labels = rotated_nms(boxes, scores, labels, 0.1, 0.05, method=2)
#         # boxes, scores, labels = rotated_soft_nms(boxes, scores, labels, 0.1, 0.05, 0.5)
#         boxes, scores, labels = rotated_diou_nms(boxes, scores, labels, 'le90', 0.1, 0.05)
#         image_file = f.name.replace('.pkl', '.png')
#         boxess = torch.cat([boxes, scores[:, None]], dim=-1).numpy()
#         labels = labels.numpy()
#         imshow_det_rbboxes(image_file, boxess, labels, show=False, out_file=image_file.replace('.png', '_diounms_det.png'))
#
