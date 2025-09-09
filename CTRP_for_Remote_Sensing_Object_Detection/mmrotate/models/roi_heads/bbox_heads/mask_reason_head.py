# -*- coding: utf-8 -*-
# @Time    : 11/16/2023
# @Author  : Peng Sun
# @FileName: mask_bbox_reason_head_test.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from mmdet.models.losses import accuracy
from mmcv.runner import BaseModule, force_fp32

from mmrotate.core import rbbox_overlaps, multiclass_nms_rotated, hbb2obb
from ...builder import ROTATED_HEADS, build_loss

from .srt import SrtMaskDecoderLayer, SrtMaskDecoder, compute_position_embeddings

from mmrotate.post_processing import multiclass_preprocess, synth_rotated

@ROTATED_HEADS.register_module()
class MaskReasonHead(BaseModule):

    def __init__(self,
                 num_classes,
                 version,
                 rel_channels,
                 srt_decoder_cfg,
                 loss_mask_cls,
                 init_cfg=None):

        super(MaskReasonHead, self).__init__(init_cfg)

        self.num_classes = num_classes  # no background class
        self.angle_version=version

        num_layers = srt_decoder_cfg.num_dec_layers
        rel_channels = rel_channels
        layer_cfg = srt_decoder_cfg.layer_cfg

        repr_dim = layer_cfg['repr_dim']
        hidden_dim = layer_cfg['hidden_dim']
        nheads = layer_cfg['nheads']
        dropout = layer_cfg['dropout']

        RelPN_mode = layer_cfg['RelPN_mode']
        with_SA = layer_cfg['with_SA']
        with_SA_pe = layer_cfg['with_SA_pe']
        with_CA = layer_cfg['with_CA']
        with_semantics_prior = layer_cfg['with_semantics_prior']
        with_spatial_prior = layer_cfg['with_spatial_prior']

        RelPN_mode = 'MLP' if with_SA == False else RelPN_mode

        self.pairwise_encoding_mode = layer_cfg['pairwise_encoding_mode']
        self.pairwise_encoding_with_normalized = layer_cfg['pairwise_encoding_with_normalized']

        if RelPN_mode == 'MLP':
            self.pairwise_encoding_with_normalized = True
            self.pairwise_encoding_mode = 'bbox_with_relative'

        if self.pairwise_encoding_with_normalized:
            if self.pairwise_encoding_mode == 'bbox':
                pairwise_encoding_dim = 24
            elif self.pairwise_encoding_mode == 'center':
                pairwise_encoding_dim = 8
            elif self.pairwise_encoding_mode == 'bbox_with_relative':
                pairwise_encoding_dim = 32
            elif self.pairwise_encoding_mode == 'center_with_relative':
                pairwise_encoding_dim = 16

        elif self.pairwise_encoding_with_normalized==False:
            pairwise_encoding_dim = 8 if self.pairwise_encoding_mode == 'center' else 20

        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        self.class_query_embed = nn.Embedding(num_classes, hidden_dim)

        # 相关网络初始化
        self.rel_feat_head = nn.Sequential(
            nn.Linear(rel_channels, 512), nn.ReLU(),
            nn.Linear(512, repr_dim), nn.ReLU())

        decoder_layer = SrtMaskDecoderLayer(
            RelPN_mode=RelPN_mode,
            with_SA=with_SA,
            with_SA_pe=with_SA_pe,
            with_CA=with_CA,
            with_semantics_prior=with_semantics_prior,
            with_spatial_prior=with_spatial_prior,
            q_dim=repr_dim,
            kv_dim=hidden_dim,
            pairwise_encoding_dim=pairwise_encoding_dim,
            ffn_interm_dim=repr_dim * 4,
            num_heads=nheads,
            dropout=dropout)

        self.decoder = SrtMaskDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            return_intermediate=True)

        self.binary_classifier = nn.Linear(repr_dim, num_classes)
        self.loss_mask_cls = build_loss(loss_mask_cls)

    @property
    def custom_activation(self):
        """The custom activation."""
        return getattr(self.loss_mask_cls, 'custom_activation', False)

    def forward(self, rel_feats_list, sub_labels_list, spatial_embeddings_list):
        '''
        Mask Reason Head, with one decoder for reasoning the category of mask objects

        Args:
            x (list[Tensor]): Used to extract relation feature.
            rel_feats_list (List[Tensor]):
            sub_labels_list (List[Tensor]): a list of subject labels in relation,
            spatial_embeddings_list (List[Tensor]):
        Returns:
        '''

        assert len(rel_feats_list) == len(sub_labels_list)

        # features: used for decoder keys/values
        features = self.class_embed.weight.unsqueeze(0).repeat(len(rel_feats_list), 1, 1)

        query_embeds = []

        for i, (rel_feat, feat) in enumerate(zip(rel_feats_list, features)):
            # if rel_feat.numel() > 0:
            query = self.rel_feat_head(rel_feat)  # relation query for decoder, shape (n, repr_dim)

            query_embeds.append(
                self.decoder(
                    queries=query.unsqueeze(1),  # (n, 1, q_dim) n: num of ho_q
                    features=feat.unsqueeze(1),
                    rel_sub_labels=sub_labels_list[i],
                    q_pos=spatial_embeddings_list[i],
                    k_pos=self.class_query_embed.weight
                ).squeeze(dim=2))

        # Concatenate queries from all images in the same batch.
        query_embeds = torch.cat(query_embeds, dim=1)  # (num_dec, \sigma{n}, q_dim)

        # num_dec: return intermediate layer
        mask_cls_score = self.binary_classifier(query_embeds)[-1]  # (num_query, q_dim)

        return mask_cls_score

    @force_fp32(apply_to=('mask_cls_score', 'mask_labels'))
    def loss(self,
             mask_cls_score,
             mask_labels,
             label_weights=None,
             reduction_override=None):

        losses = dict()
        if mask_cls_score is not None:

            if label_weights is None:
                label_weights = torch.ones_like(mask_labels)

            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            # if mask_cls_score.numel() > 0:
            loss_mask_ = self.loss_mask_cls(
                mask_cls_score,
                mask_labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            if isinstance(loss_mask_, dict):
                losses.update(loss_mask_)
            else:
                losses['loss_mask'] = loss_mask_
            if self.custom_activation:
                acc_ = self.loss_mask_cls.get_accuracy(mask_cls_score, mask_labels)
                losses.update(acc_)
            else:
                losses['reason_acc'] = accuracy(mask_cls_score, mask_labels)
        return losses

    def rel_pair_sampler(self, pos_bboxes_list, pos_labels_list, det_mask_bboxes_list, img_metas, topK):
        """
        Get the 'topk' union boxes based on the distance
            between the bboxes_pos and bboxes_masked

        Notes:
            Can the number of positive samples corresponding to each occlusion box be variable?

        Args:
            pos_bboxes_list (List[Tensor]):
                in [cx, cy, w, h, a, score] format.
            det_mask_bboxes_list (List[Tensor]):
                in [cx, cy, w, h, a] format.
            pos_labels_list (List[Tensor]):
                in shape(num_pos_gt_label, ).
            img_metas: img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            topK (int):

        Returns:

        """

        device = pos_bboxes_list[0].device

        rel_regions_list = []
        rel_pairs_list = []
        subject_labels_list = []
        spatial_embeddings_list = []

        for i, (pos_bboxes, det_mask_bboxes, pos_labels, img_meta) in enumerate(
                zip(pos_bboxes_list, det_mask_bboxes_list, pos_labels_list, img_metas)):

            pos_cen_point = pos_bboxes[:, :2]
            mask_cen_point = det_mask_bboxes[:, :2]

            distances = (pos_cen_point[:, None] -
                         mask_cen_point[None, :]).pow(2).sum(-1).sqrt()  # shape(num_pos, num_mask)
            idx = torch.argsort(distances, dim=0, descending=False)

            # Coarse relationship pairs
            # Select 'topK' the nearest detection results closest to the bbox_mask.
            rel_idx = idx[:topK, :]  # index of selected bbox (tensor): shape(num_mask, num_select_bbox)

            union_bbox_list = []
            union_region_list = []
            union_idx_list = []
            sub_label_list = []

            # 在 Faster RCNN网络中, 当 proposal的维度为 4时, 对 proposal执行补零操作.
            # pos_bboxes = torch.nn.functional.pad(pos_bboxes, (0, 1, 0, 0)) if pos_bboxes.size(1) == 4 else pos_bboxes
            # embed_mode = 'rbbox' if pos_bboxes.size(1) == 5 else 'hbbox'

            # det_mask_bboxes = det_mask_bboxes[..., :4] if embed_mode=='hbbox' else det_mask_bboxes
            # hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
            # pos_xyxys = obb2xyxy(pos_bboxes, version=self.angle_version)
            # mask_xyxys = obb2xyxy(det_mask_bboxes, version=self.angle_version)

            if pos_bboxes.size(1) == 4:
                pos_ltrb = hbb2ltrb(pos_bboxes)
                pos_bboxes = hbb2obb(pos_ltrb, version='le90')

            for m, mask_bbox in enumerate(det_mask_bboxes):
            # for m, mask_xyxy in enumerate(mask_xyxys):
                for p in range(rel_idx.size(0)):
                    union_idx = torch.tensor([rel_idx[p, m], m])
                    union_bbox = torch.stack([pos_bboxes[rel_idx[p, m]], det_mask_bboxes[m]], dim=0)
                    # 第一种情况, 错误版本的程序, 但是效果最好
                    # union_region = torch.cat([
                    #     torch.min(pos_bboxes[rel_idx[p, m]][:2], mask_bbox[:2]),
                    #     torch.max(pos_bboxes[rel_idx[p, m]][2:], mask_bbox[2:])
                    # ], dim=0)

                    # 第二种尝试
                    # union_bbox = torch.stack([pos_bboxes[rel_idx[p, m]], det_mask_bboxes[m]], dim=0)
                    # union_region = torch.cat([
                    #     torch.min(pos_xyxys[rel_idx[p, m]][:2], mask_xyxy[:2]),
                    #     torch.max(pos_xyxys[rel_idx[p, m]][2:4], mask_xyxy[2:4])
                    # ], dim=0)

                    # 第三种尝试
                    union_region = pos_bboxes[rel_idx[p, m]]

                    union_idx_list.append(union_idx.unsqueeze(0))
                    union_bbox_list.append(union_bbox.unsqueeze(0))
                    union_region_list.append(union_region.unsqueeze(0))
                    sub_label_list.append(pos_labels[rel_idx[p, m]].unsqueeze(0))

            if len(union_region_list)==0:
                union_regions = torch.ones((1, 5), device=device)
                union_idxes = torch.zeros((1, 2), dtype=torch.long)
                sub_labels = torch.zeros((1), dtype=torch.long)
                union_bboxes = torch.ones((1, 2, 5), device=device)
            else:
                union_regions = torch.cat(union_region_list, dim=0)

                union_idxes = torch.cat(union_idx_list, dim=0)
                sub_labels = torch.cat(sub_label_list, dim=0)
                union_bboxes = torch.cat(union_bbox_list, dim=0)

            spatial_embeddings = compute_position_embeddings(union_bboxes, img_meta,
                                                             self.pairwise_encoding_with_normalized,
                                                             self.pairwise_encoding_mode)

            rel_regions_list.append(union_regions)
            # rel_regions_list.append(xxyy2hbb(union_regions))
            rel_pairs_list.append(union_idxes)
            subject_labels_list.append(sub_labels)
            spatial_embeddings_list.append(spatial_embeddings)

        return rel_regions_list, rel_pairs_list, subject_labels_list, spatial_embeddings_list

    def get_targets(self,
                    gt_mask_bboxes,
                    gt_mask_labels,
                    det_mask_bboxes,
                    rel_pairs):

        # 首先获取 det_mask_labels,
        # 方式是计算 det_mask_bboxes和 gt_mask_labels的 IoU, IoU最大处的索引即为 det_mask_labels.
        det_mask_labels = []
        for gt_mask_bbox, gt_mask_label, det_mask_bbox in zip(gt_mask_bboxes, gt_mask_labels, det_mask_bboxes):

            ious = rbbox_overlaps(gt_mask_bbox, det_mask_bbox)  # (num_mask, num_gt_bboxes)
            select_gt_inds = torch.argmax(ious, dim=0)  # select mask associate with gt

            det_mask_label = gt_mask_label[select_gt_inds]
            det_mask_labels.append(det_mask_label)

        assert len(rel_pairs) == len(det_mask_labels)

        mask_label_targets_list=[]
        for rel_pair, det_mask_label in zip(rel_pairs, det_mask_labels):
            # if pairs.numel() > 0:
            label_inds = rel_pair[:, 1]

            mask_label_targets=[det_mask_label[ind] for ind in label_inds]

            mask_label_target=[]
            for ind in label_inds:
                mask_label_target.append(det_mask_label[ind])
            mask_label_target = torch.stack(mask_label_targets, dim=0)
            mask_label_targets_list.append(mask_label_target)

            # mask_label_targets_list = [mask_labels[ind] for ind in label_inds]
        mask_label_targets = torch.cat(mask_label_targets_list, dim=0)

        return mask_label_targets

    def get_mask_bboxes(self,
                        bboxes,
                        scores,
                        rel_pairs,
                        scale_factor,
                        rescale=False,
                        cfg=None):
        """
        三元组, 主语为检测出的物体, 宾语为 mask实例
        """

        if bboxes.shape[0] == 0:
            return torch.zeros((0, 6)), torch.zeros([])

        else:
            inds = rel_pairs[:, 1]
            scores = F.softmax(scores, dim=-1) if scores is not None else None
            mask_bboxes = bboxes[inds]

            # _, mask_label = torch.max(scores, dim=1)

            if rescale and mask_bboxes.size(0) > 0:
                scale_factor = mask_bboxes.new_tensor(scale_factor)
                mask_bboxes = mask_bboxes.view(mask_bboxes.size(0), -1, 5)
                mask_bboxes[..., :4] = mask_bboxes[..., :4] / scale_factor
                mask_bboxes = mask_bboxes.view(mask_bboxes.size(0), -1)

            synth_cfg = cfg.get('synth_cfg', None)
            if cfg is None:
                return mask_bboxes, scores
            else:

                a = torch.ones([scores.size()[0], 1], device=scores.device) * 0.86
                scores = torch.cat((scores, a), dim=1)

                det_mask_bboxes, det_mask_labels = multiclass_nms_rotated(
                    mask_bboxes, scores, cfg.score_thr, cfg.reason_nms, cfg.max_per_img)

                # if synth_cfg is None:
                #     det_mask_bboxes, det_mask_labels = multiclass_nms_rotated(
                #         mask_bboxes, scores, cfg.score_thr, cfg.reason_nms, cfg.max_per_img)
                #
                # else:
                #     filtered_bboxes, keep_scores, keep_labels = multiclass_preprocess(mask_bboxes, scores, cfg.score_thr)
                #     det_mask_bboxes, det_mask_labels = synth_rotated(filtered_bboxes, keep_scores, keep_labels, cfg.deepcopy())

                # 用于替换掉数值太大的 分类得分score
                # 这是一句煞笔话，不知道为啥要加他，导致结果对不齐！！！！！！
                # scores = (0.95 - 0.7) * torch.rand(det_mask_bboxes.size(0), 1) + 0.7
                # det_mask_bboxes[:, 5] = scores[:, 0]

                return det_mask_bboxes, det_mask_labels

def hbb2ltrb(hbboxes):
    x_lt = hbboxes[..., 0] - hbboxes[..., 2] * 0.5
    y_lt = hbboxes[..., 1] - hbboxes[..., 3] * 0.5
    x_rb = hbboxes[..., 0] + hbboxes[..., 2] * 0.5
    y_rb = hbboxes[..., 1] + hbboxes[..., 3] * 0.5

    ltrbs = torch.stack([x_lt,y_lt,x_rb,y_rb], dim=-1)

    return ltrbs


def xxyy2hbb(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]

    obboxes1 = torch.stack([x, y, w, h], dim=-1)
    obboxes2 = torch.stack([x, y, h, w], dim=-1)
    obboxes = torch.where((w >= h)[..., None], obboxes1, obboxes2)

    return obboxes