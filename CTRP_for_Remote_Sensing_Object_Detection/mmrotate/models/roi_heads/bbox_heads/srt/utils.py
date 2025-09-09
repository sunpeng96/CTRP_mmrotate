import math
import torch
from torch import Tensor

def compute_position_embeddings(union_bboxes, img_meta,
                                pairwise_encoding_with_normalized,
                                pairwise_encoding_mode):

    eps = 1e-10
    # if union_bboxes.numel() > 0:
    img_shape = img_meta['img_shape'][:2]

    w, h = img_shape[0], img_shape[1]
    # bbox normalization
    union_bboxes[:, :, :4] = union_bboxes[:, :, :4] / torch.tensor([w, h, w, h], device=union_bboxes.device)

    bboxes_1 = union_bboxes[:, 0, :] # ([Tensor]): shape (n, 5)
    bboxes_2 = union_bboxes[:, 1, :] # ([Tensor]): shape (n, 5)

    # compute pairwise encodings
    if pairwise_encoding_with_normalized:  # in shape(n, 20 or 32)
        pairwise_feat = compute_pairwise_encodings_with_normalized(bboxes_1, bboxes_2, img_shape, pairwise_encoding_mode)
    else:
        pairwise_feat = compute_pairwise_encodings(bboxes_1, bboxes_2, pairwise_encoding_mode)

    # compute bbox embeddings
    cxy_1 = bboxes_1[:, :2]; cxy_2 = bboxes_2[:, :2]
    c1_pe = compute_sinusoidal_pe(cxy_1[:, None], 20).squeeze(1)
    c2_pe = compute_sinusoidal_pe(cxy_2[:, None], 20).squeeze(1)

    wh_1 = bboxes_1[:, 2:4]; wh_2 = bboxes_2[:, 2:4]
    wh1_pe = compute_sinusoidal_pe(wh_1[:, None], 20).squeeze(1)
    wh2_pe = compute_sinusoidal_pe(wh_2[:, None], 20).squeeze(1)

    a_1 = torch.stack([torch.cos(bboxes_1[:, 4]), torch.sin(bboxes_1[:, 4])], dim=-1)
    a_2 = torch.stack([torch.cos(bboxes_2[:, 4]), torch.sin(bboxes_2[:, 4])], dim=-1)

    a1_pe = compute_sinusoidal_pe(a_1[:, None], 20).squeeze(1)
    a2_pe = compute_sinusoidal_pe(a_2[:, None], 20).squeeze(1)

    bbox_1_pe = torch.cat([c1_pe, wh1_pe, a1_pe], dim=-1)
    bbox_2_pe = torch.cat([c2_pe, wh2_pe, a2_pe], dim=-1)

    bbox_pe = torch.cat([bbox_1_pe, bbox_2_pe], dim=-1)
    # in shape(n, 1024) in 'hbbox' mode and shape(n, 1536) in 'rbbox' mode

    # compute centre position embeddings
    centre = (bboxes_1[:, :2] + bboxes_2[:, :2]) / 2
    centre_pe = compute_sinusoidal_pe(centre[:, None], 20).squeeze(1)  # shape (n, 1, dim)

    # compute spatial transformer embeddings
    d_cxy = torch.abs(bboxes_2[:, :2] - bboxes_1[:, :2])

    wh1_max, _ = torch.max(bboxes_1[:, 2:4], 1)
    wh1_min, _ = torch.min(bboxes_1[:, 2:4], 1)

    wh2_max, _ = torch.max(bboxes_2[:, 2:4], 1)
    wh2_min, _ = torch.min(bboxes_2[:, 2:4], 1)

    d_wh_max = (wh2_max / (wh1_max + eps)) # delta w
    d_wh_min = (wh2_min / (wh1_min + eps)) # delta h

    d_wh = torch.stack([d_wh_max, d_wh_min], dim=-1)

    d_a = bboxes_2[:, 4] - bboxes_1[:, 4]
    d_theta = torch.stack([torch.cos(d_a), torch.sin(d_a)], dim=-1)

    d_cxy_pe = compute_sinusoidal_pe(d_cxy[:, None], 20).squeeze(1)
    d_wh_pe = compute_sinusoidal_pe(d_wh[:, None], 20).squeeze(1)

    d_theta_pe = compute_sinusoidal_pe(d_theta[:, None], 20).squeeze(1)

    st_pe = torch.cat([d_cxy_pe, d_wh_pe, d_theta_pe], dim=-1)

    spatial_embeddings = {'bbox_pe': bbox_pe.unsqueeze(1),
                          'st_pe': st_pe.unsqueeze(1),
                          'centre_pe': centre_pe.unsqueeze(1),
                          'pairwise_feat': pairwise_feat.unsqueeze(1)}  # in shape(n, 16) with 'hbbox' mode and shape(n, 20) with 'rbbox' mode
    # in 'rbbox' mode: torch.Size([12, 1, 1536])
    #                  torch.Size([12, 1, 768])
    #                  torch.Size([12, 1, 256])
    #                  torch.Size([12, 1, 20])
    # in 'hbbox' mode: torch.Size([n, bs, 1024])
    #                  torch.Size([n, bs, 512])
    #                  torch.Size([n, bs, 256])
    #                  torch.Size([n, bs, 16])

    return spatial_embeddings

def compute_pairwise_encodings(bboxes_1, bboxes_2, encoding_mode):

    eps = 1e-3
    # Construct bbox encoding
    f = torch.stack([bboxes_1[:, 0], bboxes_1[:, 1], # c1_x, c1_y
                     bboxes_2[:, 0], bboxes_2[:, 1], # c2_x, c2_y
                     #-------------------------------------------
                     bboxes_1[:, 2], bboxes_1[:, 3], # w1, h1
                     bboxes_2[:, 2], bboxes_2[:, 3], # w2, h2
                     # -------------------------------------------
                     torch.sin(bboxes_1[:, 4]), torch.cos(bboxes_1[:, 4]),
                     torch.sin(bboxes_2[:, 4]), torch.cos(bboxes_2[:, 4])], dim=-1) # theta
    f[f < 0] = 1
    # Notes: 角度信息嵌入没有进行 log处理, log处理会导致 Nan.
    if encoding_mode == 'bbox':
        pairwise_feat = torch.cat([f[:, :8], torch.log(f[:, :8] + eps), f[:, 8:]], dim=-1)  # shape(n, 20)

    elif encoding_mode == 'center':
        pairwise_feat = torch.cat([f[:, :4], torch.log(f[:, :4] + eps)], dim=-1)  # shape(n, 8)

    return pairwise_feat

def compute_pairwise_encodings_with_normalized(bboxes_1, bboxes_2, img_shape, encoding_mode):
    eps = 1e-3
    h, w = img_shape

    c1_x, c1_y = bboxes_1[:, 0], bboxes_1[:, 1]
    c2_x, c2_y = bboxes_2[:, 0], bboxes_2[:, 1]

    b1_w, b1_h = bboxes_1[:, 2], bboxes_1[:, 3]
    b2_w, b2_h = bboxes_2[:, 2], bboxes_2[:, 4]

    b1_theta, b2_theta = bboxes_1[:, 4], bboxes_2[:, 4]

    d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
    d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

    # Construct bbox encoding
    f = torch.stack([
        # Relative position of box centre
        c1_x / w, c1_y / h, c2_x / w, c2_y / h,
        # Relative box width and height
        b1_w / w, b1_h / h, b2_w / w, b2_h / h,
        # Relative box angle
        torch.sin(b1_theta)+eps, torch.cos(b1_theta)+eps,
        torch.sin(b2_theta)+eps, torch.cos(b2_theta)+eps,
        # Relative distance and direction of the object w.r.t. the subject
        (c2_x > c1_x).float() * d_x,
        (c2_x < c1_x).float() * d_x,
        (c2_y > c1_y).float() * d_y,
        (c2_y < c1_y).float() * d_y,
    ], dim=-1)

    f[f < 0] = eps

    # Notes: 角度信息嵌入没有进行 log处理, log处理会导致 Nan.
    if encoding_mode == 'bbox':  # shape(n, 24)
        pairwise_feat = torch.cat([f[:, :12], torch.log(f[:, :12] + eps)], dim=-1)

    elif encoding_mode == 'bbox_with_relative':  # shape(n, 32)
        pairwise_feat = torch.cat([f, torch.log(f + eps)], dim=-1)

    elif encoding_mode == 'center':  # shape(n, 8)
        pairwise_feat = torch.cat([f[:, :4], torch.log(f[:, :4] + eps)], dim=-1)

    elif encoding_mode == 'center_with_relative':  # shape(n, 16)
        pairwise_feat = torch.cat([f[:, :4], f[:, 12:16], torch.log(f[:, :4] + eps), torch.log(f[:, 12:16] + eps)], dim=-1)  # shape(n, 8)

    return pairwise_feat


def compute_sinusoidal_pe(pos_tensor: Tensor, temperature: float = 10000.) -> Tensor:
    """
    Compute positional embeddings for points or bounding boxes

    Parameters:
    -----------
    pos_tensor: Tensor
        Coordinates of 2d points (x, y) normalised to (0, 1). The shape is (n_q, bs, 2).
    temperature: float, Default: 10000.
        The temperature parameter in sinusoidal functions.

    Returns:
    --------
    pos: Tensor
        Sinusoidal positional embeddings of shape (n_q, bs, 256).
    """
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
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

    obboxes = torch.stack([x, y, w, h], dim=-1)

    return obboxes