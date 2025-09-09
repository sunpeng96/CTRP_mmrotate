# -*- coding: utf-8 -*-
# @Time    : 03/15/2024
# @Author  : Peng Sun
# @FileName: srt_mask_oriented_rcnn.py
# @Software: PyCharm

import torch

from ..builder import ROTATED_DETECTORS
from .two_stage import RotatedTwoStageDetector

@ROTATED_DETECTORS.register_module()
class SrtOrientedRCNN(RotatedTwoStageDetector):
    """
        Implementation of `SrtOriented R-CNN for Object Detection.
    """

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None,
                 init_cfg=None
                 ):
        super(SrtOrientedRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)