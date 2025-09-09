# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import ROTATED_DETECTORS, build_head
from .two_stage import RotatedTwoStageDetector

@ROTATED_DETECTORS.register_module()
class CTRPOrientedRCNN(RotatedTwoStageDetector):
    """
        Implementation of Reasoning Oriented R-CNN for Object Detection
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None
                 ):
        super(CTRPOrientedRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            train_cfg = train_cfg if train_cfg is not None else None
            # enhance_rcnn_train_cfg = train_cfg.enhance_rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=train_cfg)
            # roi_head.update(enhance_rcnn_train_cfg=enhance_rcnn_train_cfg)

            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

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
        num_imgs = len(img)
        roi_outs = self.roi_head.forward_dummy(x, proposals, num_imgs)
        outs = outs + (roi_outs, )

        return outs