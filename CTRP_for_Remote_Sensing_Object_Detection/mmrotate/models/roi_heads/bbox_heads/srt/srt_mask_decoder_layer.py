# -*- coding: utf-8 -*-
# @Time    : 11/21/2023
# @Author  : Peng Sun
# @FileName: _bbox_forward_train_test.py
# @Software: PyCharm

import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .attention import MultiheadAttention
from typing import List, Optional, Callable

class MultiModalFusion(nn.Module):
    def __init__(self, fst_mod_size, scd_mod_size, repr_size):
        super().__init__()
        self.fc1 = nn.Linear(fst_mod_size, repr_size)
        self.fc2 = nn.Linear(scd_mod_size, repr_size)
        self.ln1 = nn.LayerNorm(repr_size)
        self.ln2 = nn.LayerNorm(repr_size)

        mlp = []
        repr_size = [2 * repr_size, int(repr_size * 1.5), repr_size]
        for d_in, d_out in zip(repr_size[:-1], repr_size[1:]):
            mlp.append(nn.Linear(d_in, d_out))
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.ln1(self.fc1(x))
        y = self.ln2(self.fc2(y))
        z = F.relu(torch.cat([x, y], dim=-1))
        z = self.mlp(z)
        return z

class SrtMaskDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate=True):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(decoder_layer.q_dim)
        self.return_intermediate = return_intermediate

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                queries,
                features,
                rel_sub_labels,
                q_attn_mask: Optional[Tensor] = None,
                qk_attn_mask: Optional[Tensor] = None,
                q_padding_mask: Optional[Tensor] = None,
                kv_padding_mask: Optional[Tensor] = None,
                q_pos: Optional[Tensor] = None,
                k_pos: Optional[Tensor] = None,
        ):
        # Add support for zero layers
        if self.num_layers == 0:
            return queries.unsqueeze(0)
        # Explicitly handle zero-size queries
        if queries.numel() == 0:
            rp = self.num_layers if self.return_intermediate else 1
            return queries.unsqueeze(0).repeat(rp, 1, 1, 1)

        output = queries
        intermediate = []

        for layer in self.layers:
            output = layer(
                queries=output,
                features=features,
                rel_sub_labels=rel_sub_labels,
                q_attn_mask=q_attn_mask,
                qk_attn_mask=qk_attn_mask,
                q_padding_mask=q_padding_mask,
                kv_padding_mask=kv_padding_mask,
                q_pos=q_pos, k_pos=k_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.return_intermediate:
            output = torch.stack(intermediate)
        else:
            output = self.norm(output).unsqueeze(0)
        return output

class SrtMaskDecoderLayer(nn.Module):
    def __init__(self,
                 RelPN_mode,
                 with_SA,
                 with_SA_pe,
                 with_CA,
                 with_semantics_prior,
                 with_spatial_prior,
                 q_dim,
                 kv_dim,
                 pairwise_encoding_dim,
                 num_heads,
                 ffn_interm_dim,
                 dropout=0.1):
        super().__init__()
        self.RelPN_mode = RelPN_mode
        self.with_SA = with_SA
        self.with_SA_pe = with_SA_pe
        self.with_CA = with_CA
        self.with_semantics_prior = with_semantics_prior
        self.with_spatial_prior = with_spatial_prior
        self.q_dim = q_dim  # repr_dim=384
        self.kv_dim = kv_dim  # hidden_dim=256
        self.pairwise_encoding_dim = pairwise_encoding_dim  # 20 or 32
        self.num_heads = num_heads
        self.dropout = dropout
        self.ffn_interm_dim = ffn_interm_dim

        # Add the missing linear projections for self-attention
        self.q_attn_q_proj = nn.Linear(q_dim, q_dim)
        self.q_attn_k_proj = nn.Linear(q_dim, q_dim)
        self.q_attn_v_proj = nn.Linear(q_dim, q_dim)
        # Each scalar is mapped to a vector of shape kv_dim // 2.
        # For a box pair, the dimension is 8 * (kv_dim // 2).
        self.q_attn_qpos_proj = nn.Linear(kv_dim, q_dim)
        self.q_attn_kpos_proj = nn.Linear(kv_dim, q_dim)

        # Linear projections on qkv have been removed in this custom layer.

        self.q_attn = MultiheadAttention(q_dim, num_heads, dropout=dropout)

        self.pairwise_head = nn.Sequential(
            nn.Linear(self.pairwise_encoding_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, q_dim), nn.ReLU(),)

        # Add the missing linear projections for cross-attention
        # 在self-attention中
        Eq_dim = q_dim+kv_dim if self.with_semantics_prior else q_dim
        self.mmf = MultiModalFusion(Eq_dim, q_dim, Eq_dim)
        self.qk_attn_q_proj = nn.Linear(Eq_dim, q_dim)

        self.qk_attn_k_proj = nn.Linear(kv_dim, q_dim)
        self.qk_attn_v_proj = nn.Linear(kv_dim, q_dim)

        self.qk_attn_kpos_proj = nn.Linear(kv_dim, q_dim)

        self.qk_attn_qpos_proj = nn.Linear(kv_dim * 3, q_dim)  # 到底需要几个 kv_dim, 需要把位置嵌入部分写好了, 后续斟酌

        self.qk_attn = MultiheadAttention(q_dim * 2, num_heads, dropout=dropout, vdim=q_dim)  # q_dim * 2 有一个 q_dim是位置嵌入

        # if RelPN_mode == 'RelEncoder':  # MLP or RelEncoder
        #     self.qk_attn = MultiheadAttention(q_dim * 2, num_heads, dropout=dropout, vdim=q_dim) #  q_dim * 2 有一个 q_dim是位置嵌入
        # else:
        #     self.qk_attn = MultiheadAttention(q_dim, num_heads, dropout=dropout, vdim=q_dim)

        self.ffn = nn.Sequential(
            nn.Linear(q_dim, ffn_interm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_interm_dim, q_dim)
        )
        self.ln1 = nn.LayerNorm(q_dim)
        self.ln2 = nn.LayerNorm(q_dim)
        self.ln3 = nn.LayerNorm(q_dim)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.dp3 = nn.Dropout(dropout)

    def forward(self,
                queries: Tensor,
                features: Tensor,
                rel_sub_labels: Tensor,
                q_pos: Tensor,
                k_pos: Tensor,
                q_attn_mask: Optional[Tensor] = None,
                qk_attn_mask: Optional[Tensor] = None,
                q_padding_mask: Optional[Tensor] = None,
                kv_padding_mask: Optional[Tensor] = None,
                ):
        '''

        Args:
            queries:
            features:
            rel_sub_labels:
            q_pos:
            k_pos:
            q_attn_mask:
            qk_attn_mask:
            q_padding_mask:
            kv_padding_mask:

        Returns:
        '''

        q = self.q_attn_q_proj(queries)
        k = self.q_attn_k_proj(queries)
        v = self.q_attn_v_proj(queries)

        q_p = self.q_attn_qpos_proj(q_pos['centre_pe'])
        k_p = self.q_attn_kpos_proj(q_pos['centre_pe'])
        # q_pos['centre_pe']: shape(n, 256)

        if self.with_SA_pe:
            q = q + q_p
            k = k + k_p

        if self.with_SA:
            q_attn = self.q_attn(
                q, k, value=v, attn_mask=q_attn_mask,
                key_padding_mask=q_padding_mask
            )[0]
        else:
            q_attn = self.q_attn(
                q, k, value=v, attn_mask=q_attn_mask
            )[0]

        queries = self.ln1(queries + self.dp1(q_attn))  # in shape [n, 1, 384]

        # initial version
        # class_rel_queries=[]
        # for i, (rel_sub_label, query) in enumerate(zip(rel_sub_labels, queries)):
        #     class_rel_query = torch.cat([features[rel_sub_label], query], dim=-1)  # label_embeds and queries correspond one to one
        #     class_rel_queries.append(class_rel_query)
        # class_rel_queries = torch.stack(class_rel_queries, dim=0)  # in shape [n, 1, 640]
        # --------------------------------

        # refine version
        features_repeat = features.unsqueeze(0).repeat(rel_sub_labels.size()[0], 1, 1, 1)
        category_embeding = features_repeat[torch.arange(features_repeat.shape[0]), rel_sub_labels]  # in shape [n, 1, 256]
        # --------------------------------

        # construct spatial explicit queries for cross-attention
        pairwise_spatial_encoding = self.pairwise_head(q_pos['pairwise_feat'])  # in shape [n, 1, 384]

        explicit_queries = queries
        if self.with_semantics_prior:
            explicit_queries = torch.cat([category_embeding, queries], dim=-1)  # label_embeds and queries correspond one to one

        if self.with_spatial_prior:
            explicit_queries = self.mmf(explicit_queries, pairwise_spatial_encoding)  # in shape [n, 1, 640]

        # construct class explicit queries for cross-attention
        q = self.qk_attn_q_proj(explicit_queries)
        k = self.qk_attn_k_proj(features)
        v = self.qk_attn_v_proj(features)

        q_p = self.qk_attn_qpos_proj(q_pos['st_pe'])
        k_p = self.qk_attn_kpos_proj(k_pos)

        n_q, bs, _ = q.shape
        q = q.view(n_q, bs, self.num_heads, self.q_dim // self.num_heads)
        q_p = q_p.view(n_q, bs, self.num_heads, self.q_dim // self.num_heads)
        q = torch.cat([q, q_p], dim=3).view(n_q, bs, self.q_dim * 2)

        hw, _, _ = k.shape
        k = k.view(hw, bs, self.num_heads, self.q_dim // self.num_heads)
        k_p = k_p.view(hw, bs, self.num_heads, self.q_dim // self.num_heads)
        k = torch.cat([k, k_p], dim=3).view(hw, bs, self.q_dim * 2)

        if self.with_CA:
            qk_attn = self.qk_attn(
                query=q, key=k, value=v, attn_mask=qk_attn_mask,
                key_padding_mask=kv_padding_mask
            )[0]
        else:
            qk_attn = self.qk_attn(
                query=q, key=k, value=v
            )[0]

        queries = self.ln2(queries + self.dp2(qk_attn))
        queries = self.ln3(queries + self.dp3(self.ffn(queries)))

        return queries