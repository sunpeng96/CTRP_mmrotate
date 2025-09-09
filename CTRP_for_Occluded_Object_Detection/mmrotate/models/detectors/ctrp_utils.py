# -*- coding: utf-8 -*-
# @Time    : 09/05/2025
# @Author  : Peng Sun
# @FileName: ctrp_utils.py
# @Software: PyCharm

import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..utils import ConvModule

class CTRPModule(nn.Module):
    def __init__(self,
                 adj_gt='mmrotate/models/detectors/graph/dota_graph_r.pkl',
                 bbox_head_cfg=None,
                 normalize=None,
                 graph_out_channels=256):
        super(CTRPModule, self).__init__()

        self.retina_cls_fc = nn.Linear(2304, 1024)

        self.normalize = normalize
        self.with_bias = normalize is None
        self.relu = nn.ReLU(inplace=True)

        # init cmp attention
        self.init_cmp_attention(bbox_head_cfg)

        if adj_gt is not None:
            self.adj_gt = pickle.load(open(adj_gt, 'rb'))
            self.adj_gt = np.float32(self.adj_gt)
            self.adj_gt = nn.Parameter(torch.from_numpy(self.adj_gt), requires_grad=False)
        self.graph_out_channels = graph_out_channels
        self.graph_weight_fc = nn.Linear(bbox_head_cfg['in_channels'] * 4 + 1, self.graph_out_channels)

    def init_cmp_attention(self, bbox_head_cfg):
        self.cmp_attention = nn.ModuleList()
        self.cmp_attention.append(
            ConvModule(1024, 1024 // 16,
                       3, stride=2, padding=1, normalize=self.normalize, bias=self.with_bias))
        self.cmp_attention.append(
            nn.Linear(1024 // 16, bbox_head_cfg['in_channels']*4 + 1))

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

    def forward(self, x, cls_scores, bbox_head, img_metas):

        # precmp attention
        base_feats = self.precmp_attention(x)

        # 1.build global semantic pool
        global_semantic_pool = torch.cat((self.retina_cls_fc(bbox_head.retina_cls.weight.reshape(135, -1)),
                                          bbox_head.retina_cls.bias.unsqueeze(1)), 1).detach()
        # 2.compute graph attention
        attention_map = nn.Softmax(1)(torch.mm(base_feats, torch.transpose(global_semantic_pool, 0, 1)))
        # 3.adaptive global reasoning
        alpha_em = attention_map.unsqueeze(-1) * torch.mm(self.adj_gt[:15, :15], global_semantic_pool.reshape(15, 9, -1).reshape(15, -1)).reshape(15, 9, -1).reshape(135, -1).unsqueeze(0)
        alpha_em = alpha_em.view(-1, global_semantic_pool.size(-1))
        alpha_em = self.graph_weight_fc(alpha_em)
        alpha_em = self.relu(alpha_em)

        # enhanced_feat = torch.mm(nn.Softmax(1)(cls_score), alpha_em)
        n_classes = bbox_head.retina_cls.weight.size(0)

        assert len(cls_scores) == len(x)
        out=[]
        for i in range(len(cls_scores)):

            cls_prob = nn.Softmax(1)(cls_scores[i]).view(len(img_metas), -1, n_classes)
            enhanced_feat = torch.bmm(cls_prob, alpha_em.view(len(img_metas), -1, self.graph_out_channels))
            enhanced_feat = enhanced_feat.view(-1, self.graph_out_channels, x[i].size(2), x[i].size(2))

            out.append(enhanced_feat)

        return out






