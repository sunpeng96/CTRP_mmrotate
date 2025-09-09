_base_ = ['./ctrp_rotated_faster_rcnn_r50_fpn_1x_occluded_dota_le90.py']

work_dir = './work_dirs/occluded_dota/srt_mask_rotated_faster_rcnn_r101_fpn_250423'
eval_mode = 'sota'  # 'ablation' or 'sota'

# Hyper-parameters set
mask_mode = True  # 用于控制是否使用实例遮挡 Model和遮挡推理 Model, 二者是同步的, 要不都用, 要不都不用.
mask_ratio = 0.1
num_dec_layers = 2
rel_pair_topK = 5

# RelPN Mode
RelPN_mode = 'RelEncoder'  # MLP or RelEncoder

# S2REncoder design and Query components
with_SA = True  # True or False
with_SA_pe = True
nheads = 8
with_semantics_prior = False
with_spatial_prior = True
# Pairwise Encoding Hyperparameters
pairwise_encoding_mode = 'bbox'  # bbox, bbox_with_relative, center, center_with_relative
with_normalized = False

# S2RDecoder design
with_CA = True  # True or False

find_unused_parameters = False if with_spatial_prior and with_SA else True
angle_version = 'le90'
model = dict(
    type='CTRPMaskRotatedFasterRCNN',
    use_mask_img=mask_mode,  # 用于指示是否使用实例遮挡模块
    mask_ratio=mask_ratio,
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    roi_head=dict(
        type='CRTrMaskRotatedStandardRoIHead',
        with_mask_reason=mask_mode,  # with_mask_reason: 用于指示是否使用推理模块
        rel_pair_topK=rel_pair_topK,
        version=angle_version,
        mask_reason_head=dict(
            type='MaskReasonHead',
            num_classes=15,
            version=angle_version,
            rel_channels=1024,  # Channels of relation feature
            srt_decoder_cfg=dict(
                num_dec_layers=num_dec_layers,
                layer_cfg=dict(
                    repr_dim=384,
                    hidden_dim=256,
                    nheads=nheads,
                    dropout=0.1,
                    # Network structure settings
                    RelPN_mode=RelPN_mode,
                    with_SA=with_SA,
                    with_SA_pe=with_SA_pe,
                    with_CA=with_CA,
                    with_semantics_prior=with_semantics_prior,
                    with_spatial_prior=with_spatial_prior,
                    pairwise_encoding_mode=pairwise_encoding_mode,
                    pairwise_encoding_with_normalized=with_normalized),  # 20 or 32
                return_intermediate=True)))
)

evaluation = dict(interval=1, metric='mAP', eval_mode=eval_mode)  # 'ablation' or 'sota'

optimizer = dict(lr=0.0025*2*2)  # 0.0025 * workers_per_gpu * num_GPU
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './work_dirs/dota/rotated_faster_rcnn_r101_fpn_250422/latest.pth'
resume_from = None
work_dir = work_dir
workflow = [('train', 1)]