# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'
from mmrotate.structures.bbox.bbox_overlaps import rbbox_overlaps
import torch

import warnings
import numpy as np


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]), len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print('Error. Length of boxes arrays not equal to length of labels array: {} != {}'.format(len(boxes[t]), len(labels[t])))
            exit()

        unique_label = np.unique(labels[t])
        for j in unique_label:
            idx = labels[t] == j
            n = len(labels[t][idx])
            box_ = np.concatenate([np.full((n, 1), j),
                                   (scores[t][idx] * weights[t]).reshape(-1, 1),
                                   np.full((n, 1), weights[t]),
                                   np.full((n, 1), t),
                                   boxes[t][idx, :]], axis=1)
            new_boxes[j] = box_[box_[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, weight, model index, x1, y1, x2, y2)
    """

    box = np.zeros(9, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:] += (b[1] * b[4:])
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type in ('avg', 'box_and_model_avg', 'absent_model_aware_avg'):
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2] = w
    box[3] = -1 # model index field is retained for consistency but is not used.
    box[4:] /= conf
    return box


def find_matching_box_fast(boxes_list, new_box, match_iou):
    """
        Reimplementation of find_matching_box with numpy instead of loops. Gives significant speed up for larger arrays
        (~100x). This was previously the bottleneck since the function is called for every entry in the array.
    """
    def rbox_iou_array(boxes, new_box):
        if isinstance(boxes, np.ndarray):
            boxes_c = torch.tensor(boxes).float()
        elif isinstance(boxes, torch.Tensor):
            boxes_c = boxes.float()
        else:
            raise TypeError
        if isinstance(new_box, np.ndarray):
            new_box_c = torch.tensor(new_box).float()
        elif isinstance(new_box, torch.Tensor):
            new_box_c = new_box.float()
        else:
            raise TypeError
        return rbbox_overlaps(boxes_c, new_box_c).reshape(-1).numpy()

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    # boxes = np.array(boxes_list)
    boxes = boxes_list

    # ious = bb_iou_array(boxes[:, 4:], new_box[4:])
    ious = rbox_iou_array(boxes[:, 4:], new_box[None, 4:])

    ious[boxes[:, 0] != new_box[0]] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou


def weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=None,
        iou_thr=0.55,
        skip_box_thr=0.0,
        conf_type='avg',
        allows_overflow=False
):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes.
        'avg': average value,
        'max': maximum value,
        'box_and_model_avg': box and model wise hybrid weighted average,
        'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        print('Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 5)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0, 9))

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box_fast(weighted_boxes, boxes[j], iou_thr)

            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].copy()))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = new_boxes[i]
            if conf_type == 'box_and_model_avg':
                clustered_boxes = np.array(clustered_boxes)
                # weighted average for boxes
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weighted_boxes[i, 2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * clustered_boxes[idx, 2].sum() / weights.sum()
            elif conf_type == 'absent_model_aware_avg':
                clustered_boxes = np.array(clustered_boxes)
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / (weighted_boxes[i, 2] + weights[mask].sum())
            elif conf_type == 'max':
                weighted_boxes[i, 1] = weighted_boxes[i, 1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(len(weights), len(clustered_boxes)) / weights.sum()
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weights.sum()
        overall_boxes.append(weighted_boxes)
    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels


# if __name__ == '__main__':
#     import os
#     import pickle
#     import torch
#     from mmrotate.core.visualization.image import imshow_det_rbboxes
#     from ensemble_rboxes_nms import rotated_nms
#
#     from mmrotate.datasets.dota import DOTADataset
#     import time
#     dirs = '/workspace/mmrotate-0.3.2/work_dirs/CGR/test/rotated_retinanet_obb_r50_fpn_1x_dota_le90_cgr_test/image/'
#     # dirs = r'/workspace/mmrotate-0.3.2/work_dirs/CGR/synthesis/s2anet/images/'
#     # pkl_file = 'P1642__1024__824___0.pkl'
#     pkl_file = 'P0016__1024__0___410.pkl'
#     # pkl_file = 'P1407__1024__0___0.pkl'
#     with open(os.path.join(dirs, pkl_file), 'rb') as f:
#         dets = pickle.load(f)
#     pass
#     result = weighted_boxes_fusion([dets['boxes'].numpy()],
#                                    [dets['scores'].numpy()],
#                                    [dets['labels'].numpy()])