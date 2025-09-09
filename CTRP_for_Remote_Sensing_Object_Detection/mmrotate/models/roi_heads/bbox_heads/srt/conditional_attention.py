# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Sequence, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn.bricks.drop import Dropout

from mmrotate.utils import OptMultiConfig

def nlc_to_nchw(x: Tensor, hw_shape: Sequence[int]) -> Tensor:
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len does not match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


def coordinate_to_encoding(coord_tensor: Tensor,
                           num_feats: int = 128,
                           temperature: int = 10000,
                           scale: float = 2 * math.pi):
    """Convert coordinate tensor to positional encoding.

    Args:
        coord_tensor (Tensor): Coordinate tensor to be converted to
            positional encoding. With the last dimension as 2 or 4.
        num_feats (int, optional): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value. Defaults to 128.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
    Returns:
        Tensor: Returned encoded positional tensor.
    """
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=coord_tensor.device)
    dim_t = temperature**(2 * (dim_t // 2) / num_feats)
    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                        dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                        dim=-1).flatten(2)

    if coord_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=-1)

    elif coord_tensor.size(-1) == 4:
        w_embed = coord_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()),
                            dim=-1).flatten(2)

        h_embed = coord_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()),
                            dim=-1).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)

    elif coord_tensor.size(-1) == 5:  # 对于旋转目标的改进
        w_embed = coord_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()),
                            dim=-1).flatten(2)

        h_embed = coord_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()),
                            dim=-1).flatten(2)

        theta_embed = coord_tensor[..., 4] * scale
        pos_theta = theta_embed[..., None] / dim_t
        pos_theta = torch.stack((pos_theta[..., 0::2].sin(), pos_theta[..., 1::2].cos()),
                            dim=-1).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h, pos_theta), dim=-1)

    else:
        raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
            coord_tensor.size(-1)))
    return pos

def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse function of sigmoid, has the same
        shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class MLP(BaseModule):
    """Very simple multi-layer perceptron (also called FFN) with relu. Mostly
    used in DETR series detectors.

    Args:
        input_dim (int): Feature dim of the input tensor.
        hidden_dim (int): Feature dim of the hidden layer.
        output_dim (int): Feature dim of the output tensor.
        num_layers (int): Number of FFN layers. As the last
            layer of MLP only contains FFN (Linear).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)

        self.layers = ModuleList(
            Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of MLP.

        Args:
            x (Tensor): The input feature, has shape
                (num_queries, bs, input_dim).
        Returns:
            Tensor: The output feature, has shape
                (num_queries, bs, output_dim).
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ConditionalAttention(BaseModule):
    """A wrapper of conditional attention, dropout and residual connection.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop: A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        cross_attn (bool): Whether the attention module is for cross attention.
            Default: False
        keep_query_pos (bool): Whether to transform query_pos before cross
            attention.
            Default: False.
        batch_first (bool): When it is True, Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default: True.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 cross_attn: bool = False,
                 keep_query_pos: bool = False,
                 batch_first: bool = True,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        assert batch_first is True, 'Set `batch_first`\
        to False is NOT supported in ConditionalAttention. \
        First dimension of all DETRs in mmdet is `batch`, \
        please set `batch_first` to True.'

        self.cross_attn = cross_attn
        self.keep_query_pos = keep_query_pos
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.attn_drop = Dropout(attn_drop)
        self.proj_drop = Dropout(proj_drop)

        self._init_layers()

    def _init_layers(self):
        """Initialize layers for qkv projection."""
        embed_dims = self.embed_dims
        self.qcontent_proj = Linear(embed_dims, embed_dims)
        self.qpos_proj = Linear(embed_dims, embed_dims)
        self.kcontent_proj = Linear(embed_dims, embed_dims)
        self.kpos_proj = Linear(embed_dims, embed_dims)
        self.v_proj = Linear(embed_dims, embed_dims)
        if self.cross_attn:
            self.qpos_sine_proj = Linear(int(1.5*embed_dims), embed_dims)
        self.out_proj = Linear(embed_dims, embed_dims)

        nn.init.constant_(self.out_proj.bias, 0.)

    def forward_attn(self,
                     query: Tensor,
                     key: Tensor,
                     value: Tensor,
                     attn_mask: Tensor = None,
                     key_padding_mask: Tensor = None) -> Tuple[Tensor]:
        """Forward process for `ConditionalAttention`.

        Args:
            query (Tensor): The input query with shape [bs, num_queries,
                embed_dims].
            key (Tensor): The key tensor with shape [bs, num_keys,
                embed_dims].
                If None, the `query` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tuple[Tensor]: Attention outputs of shape :math:`(N, L, E)`,
            where :math:`N` is the batch size, :math:`L` is the target
            sequence length , and :math:`E` is the embedding dimension
            `embed_dim`. Attention weights per head of shape :math:`
            (num_heads, L, S)`. where :math:`N` is batch size, :math:`L`
            is target sequence length, and :math:`S` is the source sequence
            length.
        """
        assert key.size(1) == value.size(1), \
            f'{"key, value must have the same sequence length"}'
        assert query.size(0) == key.size(0) == value.size(0), \
            f'{"batch size must be equal for query, key, value"}'
        assert query.size(2) == key.size(2), \
            f'{"q_dims, k_dims must be equal"}'
        assert value.size(2) == self.embed_dims, \
            f'{"v_dims must be equal to embed_dims"}'

        bs, tgt_len, hidden_dims = query.size()
        _, src_len, _ = key.size()
        head_dims = hidden_dims // self.num_heads
        v_head_dims = self.embed_dims // self.num_heads
        assert head_dims * self.num_heads == hidden_dims, \
            f'{"hidden_dims must be divisible by num_heads"}'
        scaling = float(head_dims)**-0.5

        q = query * scaling
        k = key
        v = value

        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or \
                   attn_mask.dtype == torch.float64 or \
                   attn_mask.dtype == torch.float16 or \
                   attn_mask.dtype == torch.uint8 or \
                   attn_mask.dtype == torch.bool, \
                   'Only float, byte, and bool types are supported for \
                    attn_mask'

            if attn_mask.dtype == torch.uint8:
                warnings.warn('Byte tensor for attn_mask is deprecated.\
                     Use bool tensor instead.')
                attn_mask = attn_mask.to(torch.bool)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(1), key.size(1)]:
                    raise RuntimeError(
                        'The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                        bs * self.num_heads,
                        query.size(1),
                        key.size(1)
                ]:
                    raise RuntimeError(
                        'The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(
                        attn_mask.dim()))
        # attn_mask's dim is 3 now.

        if key_padding_mask is not None and key_padding_mask.dtype == int:
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(bs, tgt_len, self.num_heads,
                                head_dims).permute(0, 2, 1, 3).flatten(0, 1)
        if k is not None:
            k = k.contiguous().view(bs, src_len, self.num_heads,
                                    head_dims).permute(0, 2, 1,
                                                       3).flatten(0, 1)
        if v is not None:
            v = v.contiguous().view(bs, src_len, self.num_heads,
                                    v_head_dims).permute(0, 2, 1,
                                                         3).flatten(0, 1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bs
            assert key_padding_mask.size(1) == src_len

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [
            bs * self.num_heads, tgt_len, src_len
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bs, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(
                bs * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(
            attn_output_weights -
            attn_output_weights.max(dim=-1, keepdim=True)[0],
            dim=-1)
        attn_output_weights = self.attn_drop(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(
            attn_output.size()) == [bs * self.num_heads, tgt_len, v_head_dims]
        attn_output = attn_output.view(bs, self.num_heads, tgt_len,
                                       v_head_dims).permute(0, 2, 1,
                                                            3).flatten(2)
        attn_output = self.out_proj(attn_output)

        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bs, self.num_heads,
                                                       tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / self.num_heads

    def forward(self,
                query: Tensor,
                key: Tensor,
                query_pos: Tensor = None,
                ref_sine_embed: Tensor = None,
                key_pos: Tensor = None,
                attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                is_first: bool = False) -> Tensor:
        """Forward function for `ConditionalAttention`.
        Args:
            query (Tensor): The input query with shape [bs, num_queries,
                embed_dims].
            key (Tensor): The key tensor with shape [bs, num_keys,
                embed_dims].
                If None, the `query` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query in self
                attention, with the same shape as `x`. If not None, it will
                be added to `x` before forward function.
                Defaults to None.
            query_sine_embed (Tensor): The positional encoding for query in
                cross attention, with the same shape as `x`. If not None, it
                will be added to `x` before forward function.
                Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
            is_first (bool): A indicator to tell whether the current layer
                is the first layer of the decoder.
                Defaults to False.
        Returns:
            Tensor: forwarded results with shape
            [bs, num_queries, embed_dims].
        """

        if self.cross_attn:
            q_content = self.qcontent_proj(query)
            k_content = self.kcontent_proj(key)
            v = self.v_proj(key)

            bs, nq, c = q_content.size()
            _, hw, _ = k_content.size()

            k_pos = self.kpos_proj(key_pos)
            if is_first or self.keep_query_pos:
                q_pos = self.qpos_proj(query_pos)
                q = q_content + q_pos
                k = k_content + k_pos
            else:
                q = q_content
                k = k_content
            q = q.view(bs, nq, self.num_heads, c // self.num_heads)
            query_sine_embed = self.qpos_sine_proj(ref_sine_embed)
            query_sine_embed = query_sine_embed.view(bs, nq, self.num_heads,
                                                     c // self.num_heads)
            q = torch.cat([q, query_sine_embed], dim=3).view(bs, nq, 2 * c)
            k = k.view(bs, hw, self.num_heads, c // self.num_heads)
            k_pos = k_pos.view(bs, hw, self.num_heads, c // self.num_heads)
            k = torch.cat([k, k_pos], dim=3).view(bs, hw, 2 * c)
            ca_output = self.forward_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask)[0]
            query = query + self.proj_drop(ca_output)
        else:
            q_content = self.qcontent_proj(query)
            q_pos = self.qpos_proj(query_pos)
            k_content = self.kcontent_proj(query)
            k_pos = self.kpos_proj(query_pos)
            v = self.v_proj(query)
            q = q_content if q_pos is None else q_content + q_pos
            k = k_content if k_pos is None else k_content + k_pos
            sa_output = self.forward_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask)[0]
            query = query + self.proj_drop(sa_output)

        return query