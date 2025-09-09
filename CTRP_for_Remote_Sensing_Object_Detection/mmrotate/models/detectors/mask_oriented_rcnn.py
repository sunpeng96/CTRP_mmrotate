# -*- coding: utf-8 -*-
# @Time    : 01/27/2024
# @Author  : Peng Sun
# @FileName: mask_oriented_rcnn.py
# @Software: PyCharm

import torch
from mmcv.image import tensor2imgs

from .srt_utils import mask_detector
from ..builder import ROTATED_DETECTORS
from .two_stage import RotatedTwoStageDetector

@ROTATED_DETECTORS.register_module()
class MaskOrientedRCNN(RotatedTwoStageDetector):
    """
    在 Oriented RCNN的基础上增加了一个目标遮挡的处理, 用于训练遮挡物体, 输出的权重用于SRT算法。
    """
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(MaskOrientedRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmrotate/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 6).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        det_results = self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

        # 检测 Mask类, 并将检测结果与 det_results合并;
        results = []
        image = tensor2imgs(img, **img_metas[0]['img_norm_cfg'])  # 将 Tensor的图像转换为 numpy
        for i in range(len(image)):
            mask_result = mask_detector(image[i])  # Opencv方法检测 Mask实例的位置, 得到的是 numpy格式的数据.
            det_result = det_results[i]
            det_results[i].append(mask_result)
            results.append(det_result)

        return results
