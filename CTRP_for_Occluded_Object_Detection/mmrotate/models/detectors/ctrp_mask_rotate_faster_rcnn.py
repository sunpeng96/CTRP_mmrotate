# -*- coding: utf-8 -*-
# @Time    : 03/22/2024
# @Author  : Peng Sun
# @FileName: ctrp_mask_rotate_faster_rcnn.py
# @Software: PyCharm

import mmcv
import torch
import numpy as np
from mmcv.image import tensor2imgs
from mmrotate.core import imshow_det_rbboxes, imshow_reason_rbboxes

from ..builder import ROTATED_DETECTORS
from .two_stage import RotatedTwoStageDetector

from .ctrp_mask_utils import mask_detector
from .img_random_mask import random_mask

@ROTATED_DETECTORS.register_module()
class CTRPMaskRotatedFasterRCNN(RotatedTwoStageDetector):
    """
    Implementation 'Spatial Reasoning Transformer' on 'Rotated FasterRCNN'
    for Occluded Object Detection
    """

    def __init__(self,
                 use_mask_img,
                 mask_ratio,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(CTRPMaskRotatedFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.use_mask_img = use_mask_img
        self.mask_ratio = mask_ratio
        self.roi_head_type = roi_head['type']

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box.

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # 这部分的作用是将gt中的一部分进行掩膜, 掩膜后的 gt的 label设为 15, 其类别为 'mask'
        # 返回 gt用于训练.

        rand_mask_results = dict()
        if self.use_mask_img:
            rand_mask_results, gt_bboxes, gt_labels = random_mask(img, img_metas, self.mask_ratio,
                                                                     gt_bboxes, gt_labels)
            img = rand_mask_results['mask_image']

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

        if self.use_mask_img and self.roi_head_type == 'CRTrMaskRotatedStandardRoIHead':
            # 如果是采用推理模式, 输入图像输入到 mask_detector中, 得到掩膜实例的位置信息.
            mask_proposal_list = []
            img_device = img.device
            image = tensor2imgs(img, **img_metas[0]['img_norm_cfg'])  # 将 Tensor的图像转换为 numpy
            for i in range(len(image)):
                mask_proposal = mask_detector(image[i])
                # mask_proposal = torch.tensor(mask_proposal, dtype=torch.float32, device=img_device)
                mask_proposal_list.append(torch.tensor(mask_proposal, dtype=torch.float32, device=img_device))

            gt_mask_bboxes = rand_mask_results['mask_bboxes']
            gt_mask_labels = rand_mask_results['mask_labels']

            roi_losses = self.roi_head.forward_train(x, img_metas,
                                                     proposal_list,
                                                     gt_bboxes, gt_labels,
                                                     # ----------------------------
                                                     mask_proposal_list,
                                                     gt_mask_bboxes, gt_mask_labels,
                                                     # ----------------------------
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
        else:
            roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                     gt_bboxes, gt_labels,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)

        losses.update(roi_losses)

        return losses

    # 'simple_test' 初步的写法和 'RotatedTwoStageDetector'是一致的, 如果是一致的, 则直接继承就 ok了.
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        # 基于 Opencv方法, 为 roi_head提供 mask_proposal_list, 用于 mask_reason_head的推理.
        mask_proposal_list = []
        img_device = img.device
        image = tensor2imgs(img, **img_metas[0]['img_norm_cfg'])  # 将 Tensor的图像转换为 numpy
        for i in range(len(image)):
            mask_proposal_np = mask_detector(image[i])
            # mask_proposal = torch.tensor(mask_proposal_np, dtype=torch.float32, device=img_device)
            mask_proposal_list.append(torch.tensor(mask_proposal_np, dtype=torch.float32, device=img_device))

        return self.roi_head.simple_test(
            x, img_metas, proposal_list, mask_proposal_list, rescale=rescale)

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

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    **kwargs):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (torch.Tensor): Only if not `show` or `out_file`
        """

        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]

        # 在掩膜输出情况下的显示, 将常规检测和掩膜检测结果合并
        elif isinstance(result, dict) and 'reason_result' in result:

            bbox_result = result['bbox_result'][:15]  # 检测流程输出的结果，无掩膜
            mask_result = result['bbox_result'][15]  # 被掩膜的目标位置
            reason_result = result['reason_result']  # 基于掩膜位置推理的结果

            mask_bboxes = mask_result
            renson_bboxes = np.vstack(reason_result)

            segm_result = None
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)

        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)

        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_rbboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if isinstance(result, dict) and 'reason_result' in result:
            mask_labels = np.full(mask_bboxes.shape[0], 15)
            reason_labels = [np.full(renson_bbox.shape[0], i, dtype=np.int32) for i, renson_bbox in enumerate(reason_result)]
            reason_labels = np.concatenate(reason_labels)

            img = imshow_reason_rbboxes(
                img,
                mask_bboxes, mask_labels,
                renson_bboxes, reason_labels,
                class_names=self.CLASSES,
                score_thr=score_thr,
                bbox_color=bbox_color,
                text_color=text_color,
                thickness=thickness,
                font_size=font_size,
                win_name=win_name,
                show=show,
                wait_time=wait_time,
                out_file=out_file)

        if not (show or out_file):
            return img