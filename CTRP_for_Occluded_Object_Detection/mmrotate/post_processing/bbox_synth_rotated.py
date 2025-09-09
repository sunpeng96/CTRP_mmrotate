from .ensemble_rboxes_nmw import fast_rbox_synth_method
from .ensemble import rotated_soft_nms, rotated_diou_nms, weighted_boxes_fusion
import torch
import copy
from mmcv.ops.nms import batched_nms
from mmengine.structures import InstanceData
from mmrotate.structures import RotatedBoxes
from typing import Optional, Tuple, Union
from torch import Tensor

SILENCE = True

def synth_rotated(bboxes,
                  scores,
                  labels,
                  cfg):
    """post-processing for multi-class bboxes.

    Args:
        multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (torch.Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (float): Config of NMS.
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple (dets, labels, indices (optional)): tensors of shape (k, 5), \
        (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    synth_cfg = cfg.get('synth_cfg', None)
    assert isinstance(synth_cfg, dict)
    gauss_valid_keys = ['synth_thr', 'synth_method', 'alpha', 'beta']
    check_gauss = all([key in synth_cfg for key in gauss_valid_keys])
    others_valid_keys = ['method', 'iou_thr']
    check_others = all([key in synth_cfg for key in others_valid_keys])
    if check_gauss + check_others == 0:
        raise AttributeError('Incomplete key "' + '" ,"'.join(gauss_valid_keys + others_valid_keys) + '"' +
                             '\nCurrent key: "' + '" ,"'.join(synth_cfg) + '"')
    elif check_gauss + check_others == 2:
        raise AttributeError('Cannot use 2 types of parameters at the same time.')
    elif check_gauss + check_others == 1:
        if not SILENCE:
            print('Use cfg key: "' + '" ,"'.join(synth_cfg) + '"')
    new_synth_cfg = copy.deepcopy(synth_cfg)

    if check_gauss:
        bboxes, scores, labels = fast_rbox_synth_method(bboxes, scores, labels, cfg.nms.iou_threshold,
                                                        cfg.score_thr, -1, False, **new_synth_cfg)

    elif check_others:
        method = new_synth_cfg.get('method')
        iou_thr = new_synth_cfg.get('iou_thr')
        if method == 1:
            # WBF
            bboxes, scores, labels = weighted_boxes_fusion([bboxes.cpu().numpy()],
                                                           [scores.cpu().numpy()],
                                                           [labels.cpu().numpy()],
                                                           iou_thr=iou_thr)
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            scores = torch.tensor(scores, dtype=torch.float32)
            labels = torch.tensor(labels)
        elif method == 2:
            # soft-NMS add sigma
            sigma = new_synth_cfg.get('sigma')
            bboxes, scores, labels = rotated_soft_nms(bboxes, scores, labels, iou_thr, cfg.score_thr, sigma=sigma)
        elif method == 3:
            # DIoU-NMS
            bboxes, scores, labels = rotated_diou_nms(bboxes, scores, labels, iou_thr, cfg.score_thr)
        else:
            raise NotImplementedError('Unsupport post-processing method.')
    else:
        raise NotImplementedError('Unsupport config file.')

    if (cfg.max_per_img > 0) and (scores.numel() > cfg.max_per_img):
        scores, keep = torch.topk(scores, cfg.max_per_img)
        bboxes = bboxes[keep, :]
        labels = labels[keep]

    return bboxes, scores, labels


def postprocess_rotated(bboxes,
                        scores,
                        labels,
                        cfg):
    synth_cfg = cfg.get('synth_cfg', None) # 获取'synth_cfg'的键值，如果不存在则返回None
    if synth_cfg is None:
        det_bboxes, keep_idxs = batched_nms(bboxes, scores, labels.clone(), cfg.nms)
        keep_labels = labels[keep_idxs]
        results = InstanceData()
        results.bboxes = RotatedBoxes(det_bboxes[:, :5])
        results.scores = det_bboxes[:, -1]
        results.labels = keep_labels
        if (cfg.max_per_img > 0) and (results.scores.numel() > cfg.max_per_img):
            results = results[:cfg.max_per_img]
    else:
        assert isinstance(synth_cfg, dict)
        det_bboxes, det_scores, det_labels = synth_rotated(bboxes.clone(), scores.clone(), labels.clone(), cfg)
        results = InstanceData()
        results.bboxes = RotatedBoxes(det_bboxes)
        results.scores = det_scores
        results.labels = det_labels
    return results

def multiclass_preprocess(
    multi_bboxes: Tensor,
    multi_scores: Tensor,
    score_thr: float,
    score_factors: Optional[Tensor] = None,
    # return_inds: bool = False,
    box_dim: int = 4
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (Union[:obj:`ConfigDict`, dict]): a dict that contains
            the arguments of nms operations.
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.
        box_dim (int): The dimension of boxes. Defaults to 4.

    Returns:
        Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
            (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > box_dim:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, box_dim)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, box_dim)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, box_dim)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, box_dim)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        # dets = torch.cat([bboxes, scores[:, None]], -1)
        return (torch.empty((0, 5), dtype=torch.float32, device=multi_bboxes.device),
                torch.empty((0,), dtype=torch.float32, device=multi_bboxes.device),
                torch.empty((0,), dtype=torch.int64, device=multi_bboxes.device))

    return bboxes, scores, labels
    # dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)