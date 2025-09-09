# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from mmdet.core import bbox2roi
from ..utils import ConvModule

from mmrotate.core import (rbbox2roi, build_assigner, build_sampler, obb2xyxy,
                           rbbox2result, build_bbox_coder)

from mmrotate.models.utils import DynamicGCN

# from mmrotate.models.utils import DynamicGCN
# from mmrotate.models.utils import VigReasoningModel
# from mmrotate.models.utils import RelationalReasoningModel
# from mmdet.models.utils import build_linear_layer

from ..builder import (ROTATED_HEADS, build_head, build_roi_extractor,
                       build_shared_head)
from .rotate_standard_roi_head import RotatedStandardRoIHead

@ROTATED_HEADS.register_module()
class CTRPRotatedStandardRoIHead(RotatedStandardRoIHead):
    """

    """
    def __init__(self,
                 num_stages,
                 # child_num=4,
                 # iou_threshold=0.7,
                 # child_iou_num=10,  # 8
                 # position_feats_dim=2940,
                 use_feat_enhance = None,
                 use_DRL = None,
                 temperture=3,
                 adj_gt=None,
                 graph_out_channels=1024,
                 normalize=None,
                 bbox_roi_extractor=None,
                 init_bbox_head=None,
                 refined_bbox_roi_extractor=None,
                 refined_bbox_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 version='oc',
                 shared_num_fc=2
                 ):

        super(CTRPRotatedStandardRoIHead, self).__init__(init_cfg)

        # training and test configuration init

        self.num_stages = num_stages

        self.use_feat_enhance = use_feat_enhance
        self.use_DRL = use_DRL

        self.tempe = temperture

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.version = version

        if self.train_cfg is not None:
            self.bbox_head_cfg=[self.train_cfg.init_rcnn, self.train_cfg.refined_rcnn]

        self.normalize = normalize
        self.with_bias = normalize is None
        self.relu = nn.ReLU(inplace=True)

        # init cmp attention
        if self.use_feat_enhance:
            self.init_cmp_attention(init_bbox_head)

            if adj_gt is not None:
                self.adj_gt = pickle.load(open(adj_gt, 'rb'))
                self.adj_gt = np.float32(self.adj_gt)
                self.adj_gt = nn.Parameter(torch.from_numpy(self.adj_gt), requires_grad=False)
            self.graph_out_channels = graph_out_channels
            self.graph_weight_fc = nn.Linear(init_bbox_head['in_channels'] * 4 + 1, self.graph_out_channels)

        # bbox head init
        if shared_head is not None:
            shared_head.pretrained = pretrained
            self.shared_head = build_shared_head(shared_head)

        if init_bbox_head and refined_bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, init_bbox_head, refined_bbox_roi_extractor, refined_bbox_head)

        # init assigner bbox assigner and sampler
        self.init_assigner_sampler()

        self.with_bbox = True if init_bbox_head and refined_bbox_head is not None else False
        self.with_shared_head = True if shared_head is not None else False

        self.roi_feat_size = bbox_roi_extractor.roi_layer.output_size
        in_channels = 256
        if shared_num_fc > 0:
            in_channels *= (self.roi_feat_size * self.roi_feat_size)

        num_classes = init_bbox_head.num_classes

        ## reasoning model setting
        self.dynamic_gcn = DynamicGCN(num_classes)
        self.criterion = nn.NLLLoss() # nn.NLLLoss() or nn.CrossEntropyLoss()
        # self.child_num = child_num
        # self.iou_threshold = iou_threshold
        # self.child_iou_num = child_iou_num
        # self.position_feats_dim = position_feats_dim

        ## reasoning model initialization
        # self.enhance_bbox_roi_extractor = enhance_bbox_roi_extractor
        # self.refined_bbox_head = refined_bbox_head
        # self.init_reasoning_model()
        # self.fc = nn.Linear(12544, 256)

    def init_cmp_attention(self, bbox_head):
        self.cmp_attention = nn.ModuleList()
        self.cmp_attention.append(
            ConvModule(1024, 1024 // 16,
                       3, stride=2, padding=1, normalize=self.normalize, bias=self.with_bias))
        self.cmp_attention.append(
            nn.Linear(1024 // 16, bbox_head['in_channels']*4 + 1))

    def init_bbox_head(self, bbox_roi_extractor, init_bbox_head=None, refined_bbox_roi_extractor=None, refined_bbox_head=None):
        """Initialize ``bbox_head``.

        Args:
            bbox_roi_extractor (dict): Config of ``bbox_roi_extractor``.
            bbox_head (dict): Config of ``bbox_head``.
        """

        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [bbox_roi_extractor, refined_bbox_roi_extractor]

        bbox_head = [init_bbox_head, refined_bbox_head]

        # assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stage
        assert len(bbox_roi_extractor) == len(bbox_head)

        for i in range(self.num_stages):
            self.bbox_roi_extractor.append(build_roi_extractor(bbox_roi_extractor[i]))
            self.bbox_head.append(build_head(bbox_head[i]))

        # for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
        #     self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
        #     self.bbox_head.append(build_head(head))

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None

        if self.train_cfg is not None:
            init_bbox_assigner = build_assigner(self.train_cfg.init_rcnn.assigner)
            enhance_bbox_assigner = build_assigner(self.train_cfg.refined_rcnn.assigner)

            init_bbox_sampler = build_sampler(self.train_cfg.init_rcnn.sampler, context=self)
            enhance_bbox_sampler = build_sampler(self.train_cfg.refined_rcnn.sampler, context=self)

            self.bbox_assigner = [init_bbox_assigner, enhance_bbox_assigner]
            self.bbox_sampler = [init_bbox_sampler, enhance_bbox_sampler]

    def precmp_attention(self, x):
        if len(x) > 1:
            base_feat = []
            for b_f in x[1:]:
                base_feat.append(
                    F.interpolate(b_f, scale_factor=(x[2].size(2) / b_f.size(2),
                                                     x[2].size(3) / b_f.size(3))))
            base_feat = torch.cat(base_feat, 1)
        else:
            base_feat = torch.cat(x, 1)

        for ops in self.cmp_attention:
            base_feat = ops(base_feat)
            if len(base_feat.size()) > 2:
                base_feat = base_feat.mean(3).mean(2)
            else:
                base_feat = self.relu(base_feat)

        return base_feat

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
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
            dict[str, Tensor]: a dictionary of loss components.
        """

        losses = dict()

        for i in range(self.num_stages):
            self.current_stage = i
            lw = self.train_cfg.stage_loss_weights[i]

            # add reasoning process
            enhanced_feats=None
            if self.use_feat_enhance and i > 0:
                # precmp attention
                base_feats = self.precmp_attention(x)

                # 1.build global semantic pool
                global_semantic_pool = torch.cat((self.bbox_head[0].fc_cls.weight,
                                                  self.bbox_head[0].fc_cls.bias.unsqueeze(1)), 1).detach()
                # 2.compute graph attention
                attention_map = nn.Softmax(1)(torch.mm(base_feats, torch.transpose(global_semantic_pool, 0, 1)))
                # 3.adaptive global reasoning
                alpha_em = attention_map.unsqueeze(-1) * torch.mm(self.adj_gt, global_semantic_pool).unsqueeze(0)
                alpha_em = alpha_em.view(-1, global_semantic_pool.size(-1))
                alpha_em = self.graph_weight_fc(alpha_em)
                alpha_em = self.relu(alpha_em)
                # enhanced_feat = torch.mm(nn.Softmax(1)(cls_score), alpha_em)
                n_classes = self.bbox_head[0].fc_cls.weight.size(0)
                cls_prob = nn.Softmax(1)(cls_score).view(len(img_metas), -1, n_classes)
                enhanced_feats = torch.bmm(cls_prob, alpha_em.view(len(img_metas), -1, self.graph_out_channels))
                enhanced_feats = enhanced_feats.view(-1, self.graph_out_channels)

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox:
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                for i in range(num_imgs):

                    gt_hbbox = obb2xyxy(gt_bboxes[i], self.version)
                    gt_rbbox = gt_bboxes[i]

                    gt_bbox = gt_hbbox if self.current_stage == 0 else gt_rbbox
                    assign_result = self.bbox_assigner[self.current_stage].assign(
                        proposal_list[i],
                        gt_bbox,
                        gt_bboxes_ignore[i],
                        gt_labels[i])

                    sampling_result = self.bbox_sampler[self.current_stage].sample(
                        assign_result,
                        proposal_list[i],
                        gt_bbox,
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])

                    if gt_bboxes[i].numel() == 0:
                        sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                            (0, gt_bboxes[0].size(-1))).zero_()
                    else:
                        sampling_result.pos_gt_bboxes = \
                            gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            if self.with_bbox:
                bbox_results = self._bbox_forward_train(x, sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        img_metas, self.current_stage, enhanced_feats)

                for name, value in bbox_results['loss_bbox'].items():

                    losses['s{}.{}'.format(
                        self.current_stage, name)] = (value * lw if 'loss' in name else value)

                if self.current_stage == 0 and self.use_DRL:
                    losses['gcn_loss'] = bbox_results['gcn_loss']
                cls_score = bbox_results['cls_score']
                bbox_pred = bbox_results['bbox_pred']

            # refine bboxes
            if self.current_stage < self.num_stages - 1:

                rois = bbox2roi([res.bboxes for res in sampling_results])

                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                bbox_targets = self.bbox_head[self.current_stage].get_targets(sampling_results, gt_bboxes,
                                                             gt_labels, self.bbox_head_cfg[self.current_stage])
                roi_labels = bbox_targets[0] # bbox_targets is a tuple

                with torch.no_grad():
                    proposal_list = self.bbox_head[self.current_stage].refine_rboxes_hbb(
                        rois, roi_labels, bbox_results['bbox_pred'], pos_is_gts, img_metas)

        return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, stage=None, enhanced_feat=None):

        """
        Run forward function and calculate loss for box head in training.

        Args:
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[Tensor]): list of sampling results.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                hape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """

        if stage == 0:
            rois = bbox2roi([res.bboxes for res in sampling_results])
        else:
            rois = rbbox2roi([res.bboxes for res in sampling_results])  # shape (n, 6), [batch_ind, cx, cy, w, h, a]

        bbox_results = self._bbox_forward(x, rois, stage, enhanced_feat)

        bbox_targets = self.bbox_head[stage].get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.bbox_head_cfg[stage])
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)

        if stage == 0 and self.use_DRL:
            # cls_score --> prob_init: graph embedding
            # Roi feature: gcn input --> calculate similarity
            # gcn loss: support label and gcn output
            bbox_feats = bbox_results['bbox_feats']
            cls_score = bbox_results['cls_score']
            prob_cls = F.softmax(cls_score / self.tempe)

            labels, label_weights = bbox_targets[0], bbox_targets[1]

            out_for_gcn = self.dynamic_gcn(bbox_feats, prob_cls)
            probs_log = torch.log(out_for_gcn + 1e-12)

            # cls_score or roi label from get target
            gcn_loss = label_weights * self.criterion(probs_log, labels)  # nn.CrossEntropyLoss()
            # gcn_loss = self.criterion(probs_log, labels)  # nn.NLLLoss()

            bbox_results.update(gcn_loss=gcn_loss)

        return bbox_results

    def _bbox_forward(self, x, rois, stage=None, enhanced_feat=None):
        """Box head forward function used in both training and testing.

        Args:
            x (list[Tensor]): list of multi-level img features.
            rois (list[Tensors]): list of region of interests.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        bbox_feats = self.bbox_roi_extractor[stage](
            x[:self.bbox_roi_extractor[stage].num_inputs], rois)  # shape (1024, 256, 7, 7)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred = self.bbox_head[stage](bbox_feats)

        # if self.use_feat_enhance and stage > 0:
        #     cls_score, bbox_pred = self.bbox_head[stage](
        #         bbox_feats, enhanced_feat)
        # else:
        #     cls_score, bbox_pred = self.bbox_head[stage](
        #         bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        return bbox_results

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_list (list[Tensors]): list of region proposals.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            rbbox2result(det_bboxes[i], det_labels[i],
                         self.bbox_head[-1].num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results

    def simple_test_bboxes(self,
                               x,
                               img_metas,
                               proposals,
                               rcnn_test_cfg,
                               rescale=False):
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # multi-stage process
        ms_scores = []
        rois = bbox2roi(proposals)
        for i in range(self.num_stages):
            self.current_stage = i

            enhanced_feat=None
            if self.use_feat_enhance and i > 0:
                # precmp attention
                base_feats = self.precmp_attention(x)

                # 1.build global semantic pool
                global_semantic_pool = torch.cat((self.bbox_head[0].fc_cls.weight,
                                                  self.bbox_head[0].fc_cls.bias.unsqueeze(1)), 1).detach()
                # 2.compute graph attention
                attention_map = nn.Softmax(1)(torch.mm(base_feats, torch.transpose(global_semantic_pool, 0, 1)))
                # 3.adaptive global reasoning
                alpha_em = attention_map.unsqueeze(-1) * torch.mm(self.adj_gt, global_semantic_pool).unsqueeze(0)
                alpha_em = alpha_em.view(-1, global_semantic_pool.size(-1))
                alpha_em = self.graph_weight_fc(alpha_em)
                alpha_em = self.relu(alpha_em)
                # enhanced_feat = torch.mm(nn.Softmax(1)(cls_score), alpha_em)
                n_classes = self.bbox_head[0].fc_cls.weight.size(0)
                cls_prob = nn.Softmax(1)(cls_score).view(len(img_metas), -1, n_classes)
                enhanced_feat = torch.bmm(cls_prob, alpha_em.view(len(img_metas), -1, self.graph_out_channels))
                enhanced_feat = enhanced_feat.view(-1, self.graph_out_channels)

            bbox_results = self._bbox_forward(x, rois, self.current_stage, enhanced_feat)
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']

            if self.current_stage < self.num_stages - 1:

                bbox_label = cls_score.argmax(dim=1)
                rois = self.bbox_head[self.current_stage].regress_by_class_hbb(rois, bbox_label, bbox_pred,
                                                  img_metas[self.current_stage])

        num_proposals_per_img = tuple(len(p) for p in proposals)
        ms_scores.append(cls_score)

        cls_score = sum(ms_scores) / self.num_stages
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        rcnn_test_cfg = self.test_cfg
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0,), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)

            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels

    def forward_dummy(self, x, proposals, num_imgs=None):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        img_meta = dict()
        img_meta['img_shape'] = (1024, 1024, 3)

        outs = ()
        rois = bbox2roi([proposals])

        ms_scores = []
        for i in range(self.num_stages):
            self.current_stage = i
            lw = self.train_cfg.stage_loss_weights[i]

            # add reasoning process
            enhanced_feat=None
            if self.use_feat_enhance and i > 0:
                # precmp attention
                base_feats = self.precmp_attention(x)

                # 1.build global semantic pool
                global_semantic_pool = torch.cat((self.bbox_head[0].fc_cls.weight,
                                                  self.bbox_head[0].fc_cls.bias.unsqueeze(1)), 1).detach()
                # 2.compute graph attention
                attention_map = nn.Softmax(1)(torch.mm(base_feats, torch.transpose(global_semantic_pool, 0, 1)))
                # 3.adaptive global reasoning
                alpha_em = attention_map.unsqueeze(-1) * torch.mm(self.adj_gt, global_semantic_pool).unsqueeze(0)
                alpha_em = alpha_em.view(-1, global_semantic_pool.size(-1))
                alpha_em = self.graph_weight_fc(alpha_em)
                alpha_em = self.relu(alpha_em)
                # enhanced_feat = torch.mm(nn.Softmax(1)(cls_score), alpha_em)
                n_classes = self.bbox_head[0].fc_cls.weight.size(0)

                cls_prob = nn.Softmax(1)(cls_score).view(1, -1, n_classes)
                enhanced_feat = torch.bmm(cls_prob, alpha_em.view(1, -1, self.graph_out_channels))
                enhanced_feat = enhanced_feat.view(-1, self.graph_out_channels)

            bbox_results = self._bbox_forward(x, rois, self.current_stage, enhanced_feat)

            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']

            # refine bboxes
            if self.current_stage < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)

                rois = self.bbox_head[self.current_stage].regress_by_class_hbb(rois, bbox_label, bbox_pred,
                                                                           img_meta)

            outs = outs + (cls_score, bbox_pred)

        return outs