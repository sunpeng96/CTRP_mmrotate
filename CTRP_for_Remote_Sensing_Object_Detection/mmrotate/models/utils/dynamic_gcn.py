import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import GCN

class DynamicGCN(nn.Module):

    def __init__(self, num_classes):
        super(DynamicGCN, self).__init__()

        self.classes = num_classes
        self.gcn = GCN(in_channels=self.classes, out_channels=self.classes, hidden_layers='15')
        self.fc = nn.Linear(12544, 1024)

    def set_negative_to_zero(self, W):
        return F.relu(W)

    def _get_W(self, x):
        x = (x - x.mean(dim=1).unsqueeze(1))
        norms = x.norm(dim=1)
        W = torch.mm(x, x.t()) / torch.ger(norms, norms)
        W = self.set_negative_to_zero(W.cuda())
        return W

    def forward(self, bbox_feature, cls_score):
        # 根据 bbox_feature, bbox_pred 计算相似度
        # 计算相似度需要考虑的因素包含但不限于 —— 特征、距离、尺度、指向, 各项可以分配一定的权重

        bbox_feature = bbox_feature.view(bbox_feature.shape[0], -1)

        shaped_bbox_feature = self.fc(bbox_feature)

        similarity = self._get_W(shaped_bbox_feature)

        # RoI 的预测概率作为 embedding
        # 将大于一定阈值的预测概率置为 1, 类似于 Few-shots中的 support feature
        # support: roi_label --> prob_init = 1
        # cls_score --> prob_init
        # index = [torch.arange(0, len(roi_labels), 1), roi_labels]
        # prob_init = cls_score.index_put(index, 1)
        bg_score = cls_score[:, self.classes]
        prob_init = cls_score[:, :self.classes]

        prob_gcn = self.gcn(x=prob_init, edges=similarity)

        prob_gcn = torch.cat([prob_gcn, bg_score.unsqueeze(1)], dim=-1)

        return prob_gcn