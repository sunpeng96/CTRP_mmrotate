# -*- coding: utf-8 -*-
# @Time    : 03/22/2024
# @Author  : Peng Sun
# @FileName: srt_mask_rotate_standard_roi_head.py
# @Software: PyCharm

import torch
import torch.nn as nn
import numpy as np

from mmdet.core import bbox2roi
from mmcv.utils import to_2tuple
from mmrotate.core import obb2xyxy
from mmrotate.core import rbbox2roi, rbbox2result

from ..builder import ROTATED_HEADS, build_head, build_roi_extractor
from .rotate_standard_roi_head import RotatedStandardRoIHead

@ROTATED_HEADS.register_module()
class SrtMaskRotatedStandardRoIHead(RotatedStandardRoIHead):
    """
    Rotated Faster RCNN roi head including one bbox head with 'SpatialReasoningTransformer'
    """
    def __init__(self,
                 with_mask_reason=False,
                 rel_pair_topK=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 rel_roi_extractor=None,
                 mask_reason_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 version='oc'
                 ):
        super(SrtMaskRotatedStandardRoIHead, self).__init__(bbox_roi_extractor,
                                                         bbox_head, shared_head,
                                                         train_cfg, test_cfg,
                                                         pretrained, init_cfg,
                                                         version)

        roi_feat_size = to_2tuple(bbox_head['roi_feat_size'])
        roi_feat_area = roi_feat_size[0] * roi_feat_size[1]
        self.roi_feat_dim = bbox_head['in_channels'] * roi_feat_area

        self.rel_feats_fc = nn.Linear(self.roi_feat_dim, 1024)

        self.with_mask_reason = with_mask_reason

        self.topK = rel_pair_topK #

        if self.with_mask_reason and mask_reason_head is not None:
            self.init_mask_reason_head(rel_roi_extractor, mask_reason_head)

    def init_mask_reason_head(self, rel_roi_extractor, mask_reason_head):
        """Initialize ``bbox_head``.

        Args:
            bbox_roi_extractor (dict): Config of ``bbox_roi_extractor``.
            bbox_head (dict): Config of ``bbox_head``.
        """

        self.rel_roi_extractor = build_roi_extractor(rel_roi_extractor)
        self.mask_reason_head = build_head(mask_reason_head)

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes, gt_labels,
                      # -----------------
                      mask_proposal_list=None,
                      gt_mask_bboxes=None, gt_mask_labels=None,
                      # -----------------
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task. Always
                set to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals

        if self.with_bbox:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                gt_hbboxes = obb2xyxy(gt_bboxes[i], self.version)
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_hbboxes, gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_hbboxes,
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                        (0, gt_bboxes[0].size(-1))).zero_()
                else:
                    sampling_result.pos_gt_bboxes = \
                        gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

                sampling_results.append(sampling_result)

        losses = dict()

        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        if self.with_mask_reason:
            assert self.with_bbox, '当使用_mask_reason_forward_train时, 需要bbox_head提供遮挡物体的位置.'

            rescale = False
            img_shapes = tuple(meta['img_shape'] for meta in img_metas)
            scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

            rois = bbox2roi([res.bboxes for res in sampling_results])
            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']

            num_proposals_per_img = tuple(res.bboxes.size(0) for res in sampling_results)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

            # apply bbox post-processing to each image individually
            det_bboxes = []
            det_labels = []
            for i in range(len(sampling_results)):
                if rois[i].shape[0] == 0:
                    # There is no proposal in the single image
                    det_bbox = rois[i].new_zeros(0, 5)
                    det_label = rois[i].new_zeros((0,), dtype=torch.long)
                    if self.test_cfg is None:
                        det_bbox = det_bbox[:, :4]
                        det_label = rois[i].new_zeros(
                            (0, self.bbox_head.fc_cls.out_features))

                else:
                    det_bbox, det_label = self.bbox_head.get_bboxes(
                        rois[i],
                        cls_score[i],
                        bbox_pred[i],
                        img_shapes[i],
                        scale_factors[i],
                        rescale=rescale,
                        cfg=self.test_cfg)
                det_bboxes.append(det_bbox[..., :5])
                det_labels.append(det_label)

            det_results = dict(det_bboxes=det_bboxes, det_labels=det_labels)

            mask_bbox_pred = []
            for i, (mask_proposal, gt_mask_bbox) in enumerate(zip(mask_proposal_list, gt_mask_bboxes)):
                # 是否将 gt_mask_bbox加入到训练数据中, 如果是则不需要后续的判断.
                if self.train_cfg['add_mask_gt_as_proposals']:
                    mask_proposal = torch.cat([mask_proposal[:, :5], gt_mask_bbox], dim=0)
                # 如果不将 gt_mask_bbox加入到训练数据中, 则如果 mask_proposal为空, 就将 gt_mask_bbox补充进去.
                if mask_proposal.numel() == 0:
                    mask_proposal = gt_mask_bbox
                mask_bbox_pred.append(mask_proposal)
            # mask_bbox_pred = bbox_results['det_mask_bboxes']
            mask_results = self._mask_reason_forward_train(x, img_metas, sampling_results, det_results,
                                                           mask_bbox_pred, gt_mask_bboxes, gt_mask_labels)

            losses.update(mask_results['loss_mask'])

        return losses

    def _mask_reason_forward_train(self, x, img_metas, sampling_results, det_results,
                                   mask_bbox_pred, gt_mask_bboxes, gt_mask_labels):

        # mask label prediction forward
        # data preparation for fuction (rel_pair_sample)
        det_bboxes = det_results['det_bboxes']  # pos_bboxes (List[Tensor])
        det_labels = det_results['det_labels']  # pos_labels (List[Tensor])

        pos_bboxes = []
        pos_labels = []
        for i, (det_bbox, det_label) in enumerate(zip(det_bboxes, det_labels)):
            if det_bbox.numel() == 0:
                det_bbox = sampling_results[i].pos_bboxes[0, :].unsqueeze(0)
                det_label = sampling_results[i].pos_gt_labels[0].unsqueeze(0)
            pos_bboxes.append(det_bbox)
            pos_labels.append(det_label)

        # pos_bboxes = [res.pos_bboxes for res in sampling_results]  # pos_bboxes (List[Tensor])
        # pos_labels = [res.pos_gt_labels for res in sampling_results]  # pos_labels (List[Tensor])

        #  标签是 mask的不能作为主语
        #  注: 这部分是否还需要, 现阶段采样的样本里面是没有掩膜物体的信息的.
        selected_pos_bboxes = []
        selected_pos_labels = []
        for pos_bbox, pos_label in zip(pos_bboxes, pos_labels):
            not_mask_inds = pos_label != 15
            selected_pos_bbox = pos_bbox[not_mask_inds]
            selected_pos_label = pos_label[not_mask_inds]

            selected_pos_bboxes.append(selected_pos_bbox)
            selected_pos_labels.append(selected_pos_label)

        # ----------------------------------
        rel_regions_list, rel_pairs_list, sub_labels_list, spatial_embeddings_list = \
            self.mask_reason_head.rel_pair_sampler(selected_pos_bboxes,
                                                   selected_pos_labels,
                                                   mask_bbox_pred,
                                                   img_metas,
                                                   topK=self.topK)
        # rel_pair_bboxes (List[Tensor]): Union region of subject_bbox and mask_bbox
        # subject_labels (Tensor): Subject label corresponding to rel_pair_bboxes

        # bbox_pe, st_pe, centre_pe, pairwise_feat
        rel_rois = rbbox2roi(rel_regions_list)  # Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]
        # sub_cois = label2cois(sub_labels)  # sub_cois (Tensor): categories of interest, shape (n, 2), [batch_ind, class]

        mask_bbox_results = dict()
        mask_cls_score = self._mask_reason_forward(x, rel_rois,
                                                   sub_labels_list,
                                                   spatial_embeddings_list)

        mask_label_targets = self.mask_reason_head.get_targets(gt_mask_bboxes, gt_mask_labels,
                                                               mask_bbox_pred, rel_pairs_list)

        loss_mask = self.mask_reason_head.loss(mask_cls_score, mask_label_targets)

        mask_bbox_results.update(loss_mask=loss_mask)

        return mask_bbox_results

    def _mask_reason_forward(self, x, rel_rois, sub_labels_list, spatial_embeddings_list):
        """
        Args:
            x (list[Tensor]): Used to extract relation feature.
            rel_pair_rois ([Tensor]): Used to extract relation feature for mask reason head,
                shape(num_rel_pairs, 6) in [batch_ind, cx, cy, w, h, a] format.
            subject_labels ([Tensor]): shape(num_sub_labels, 2) in [batch_ind, label] format.

        Returns:
            mask_cls_score (([Tensor]))

        Notes:
            num_rel_pairs = num_sub_labels
        """
        num_img = len(sub_labels_list)
        # Extract ROI features and convert features form [Tensor] format to (List[Tensor]) format

        rel_feats = self.rel_roi_extractor(x[:self.rel_roi_extractor.num_inputs], rel_rois)
        # rel_feats in [Tensor] to list[Tensor], num_list is batch size.

        rel_feats = self.rel_feats_fc(rel_feats.reshape(rel_feats.size(0), -1))

        rel_inds = rel_rois[:, 0]
        rel_feats_list = [rel_feats[rel_inds == i, :] for i in range(num_img)]
        # -----------------------------------

        mask_cls_scores = self.mask_reason_head(rel_feats_list, sub_labels_list, spatial_embeddings_list)
        # scores = F.softmax(
        #     mask_cls_scores, dim=-1) if mask_cls_scores is not None else None

        return mask_cls_scores

    def simple_test(self, x, img_metas, proposal_list, mask_proposal_list=None, rescale=False):

        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        # self.simple_test_bboxes: 继承 'OrientedStandardRoIHead'类中的 simple_test_bboxes方法.
        # det_bboxes, det_labels 为 NMS之后的

        # 改进的检测框分类融合, 带 mask掩膜类别, 类别有 16类;
        bbox_results = []
        for i in range(len(det_bboxes)):
            bbox_result = rbbox2result(det_bboxes[i], det_labels[i], self.bbox_head.num_classes)
            mask_bbox_result = mask_proposal_list[i].cpu().numpy()
            bbox_result.append(mask_bbox_result)
            bbox_results.append(bbox_result)

        # 初始的检测框分类融合, 类别有 15类;
        # bbox_results = [
        #     rbbox2result(det_bboxes[i], det_labels[i],
        #                  self.bbox_head.num_classes)
        #     for i in range(len(det_bboxes))
        # ]  # bbox_results (List(list(ndarray))), batch_size=1, num_class=15, (n, 6)

        # 筛选出类别不是 Mask的检测框和标签, 作为关系推理的主语;
        # 筛选出类别为 Mask的检测框, 将其输入到 SRT中做推理.
        # det_mask_bboxes = []
        # pos_bboxes = []
        # pos_labels = []
        # for det_bbox, det_label in zip(det_bboxes, det_labels):
        #     mask_inds = det_label == 15
        #     det_mask_bboxes.append(det_bbox[mask_inds, :5])
        #     pos_bboxes.append(det_bbox[~mask_inds, :5])
        #     pos_labels.append(det_label[~mask_inds])

        det_mask_bboxes, det_mask_labels = self.simple_test_mask_reason(
            x, img_metas, det_bboxes, det_labels, mask_proposal_list, self.test_cfg, rescale=rescale)

        reason_results = [
            mask_rbbox2result(det_mask_bboxes[i], det_mask_labels[i], self.mask_reason_head.num_classes)
            for i in range(len(det_mask_bboxes))
        ]
        # 预计要写一个 'mask2result' function.
        # Notes: 将 mask_bboxes处理成和 bbox_results一种格式的数据, 第一版的程序，仅做分类的准确性判断.
        # 后续版本的程序，mask部分的结果验证就和目标检测是一致的了.
        if self.with_mask_reason:
            # results = {'bbox_results': bbox_results,
            #            'reason_results': reason_results}
            results = []
            for bbox_result, reason_result in zip(bbox_results, reason_results):
                result = {'bbox_result': bbox_result,
                          'reason_result': reason_result}
                results.append(result)

        else:
            results = bbox_results

        return results
        # return bbox_results

    def simple_test_mask_reason(self, x, img_metas,
                                pos_bboxes, pos_labels,
                                mask_proposals, rcnn_test_cfg,
                                rescale=False):
        '''

        Args:
            x:
            img_metas:
            pos_bboxes:
            pos_labels:
            mask_bboxes:

        Returns:
            mask_bboxes (List[Tensor]):
            mask_labels (List[Tensor]):
        '''

        # num_proposals_per_img = tuple(len(p) for p in pos_bboxes)

        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        mask_bbox_list = []
        mask_cls_score_list = []
        rel_pairs_list = [None, None]
        for pos_bbox, pos_label, mask_proposal, img_meta in zip(pos_bboxes, pos_labels, mask_proposals, img_metas):
            if pos_bbox.numel() != 0 and mask_proposal.numel() != 0:
                # 1. 构造 Relation rois
                rel_regions_list, rel_pairs_list, sub_labels_list, spatial_embeddings_list = \
                    self.mask_reason_head.rel_pair_sampler([pos_bbox[:, :5]], [pos_label], [mask_proposal[:, :5]],
                                                           [img_meta], topK=self.topK)
                # rel_pair_bboxes (List[Tensor]): Union region of subject_bbox and mask_bbox
                # subject_labels (Tensor): Subject label corresponding to rel_pair_bboxes
                # spatial_embeddings_list (dict[List[Tensor]]):bbox_pe, st_pe, centre_pe, pairwise_feat
                rel_rois = rbbox2roi(rel_regions_list)  # Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]

                # 2. 根据关系推理出被遮挡物体的得分. mask_cls_score ([Tensor]): shape (n, 15)
                mask_cls_score = self._mask_reason_forward(x, rel_rois, sub_labels_list, spatial_embeddings_list)
                # Notes: simple_test 程序中的mask_cls_score需要根据batch size分开, 详情参考 simple_test_bbox_reason.

                # 3. 选出 mask_cls_score的最大值的索引, 计算出被遮挡的类别.
                # det_mask_label = torch.argmax(mask_cls_score, dim=1)  # 可能用到 torch.argmax() 函数, 后期再测试喽.
                # det_mask_bbox_list.append(det_mask_bbox)
                # det_mask_cls_score_list.append(det_mask_cls_score)

            # 如果没有 det_mask_bboxes, 则补零作为输出.
            else:
                mask_proposal = np.zeros((0, 5), dtype=np.float32)
                mask_cls_score = np.zeros((0, 15), dtype=np.int64)

            mask_bbox_list.append(mask_proposal[:, :5])
            mask_cls_score_list.append(mask_cls_score)

        # det_mask_bboxes = det_mask_bboxes.split(num_proposals_per_img, 0)
        # det_mask_labels = det_mask_labels.split(num_proposals_per_img, 0)

        # 3. 根据检测出的掩膜的 bbox信息和分类得分信息得到检测出掩膜位置和对应的标签
        det_mask_bboxes = []
        det_mask_labels = []
        for i in range(len(mask_proposals)):
            det_mask_bbox, det_mask_label = self.mask_reason_head.get_mask_bboxes(
                mask_bbox_list[i],
                mask_cls_score_list[i],
                rel_pairs_list[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)

            det_mask_bboxes.append(det_mask_bbox)
            det_mask_labels.append(det_mask_label)

        return det_mask_bboxes, det_mask_labels

def mask2result(bboxes, scores, rel_pairs=None):

    if bboxes.shape[0] == 0:
        return torch.zeros((0, 5)), torch.zeros([])

    else:
        mask_label_list = []
        mask_bbox_list = []
        for i in range(bboxes.shape[0]):
            inds = rel_pairs[:, 1] == i

            mask_scores = scores[inds]
            _, mask_label = torch.max(mask_scores, dim=1)

            mask_bbox = [bboxes[i] for j in range(len(mask_scores))]
            mask_bbox = torch.stack(mask_bbox, dim=0)

            mask_label_list.append(mask_label)
            mask_bbox_list.append(mask_bbox)

        mask_labels = torch.stack(mask_label_list, dim=0)
        mask_bboxes = torch.stack(mask_bbox_list, dim=0)

        return mask_bboxes, mask_labels

def mask_rbbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): shape (n, 6)
        labels (torch.Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """

    if bboxes.numel() == 0:
        return [np.zeros((0, 6), dtype=np.float32) for _ in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()

        return [bboxes[labels == i, :] for i in range(num_classes)]