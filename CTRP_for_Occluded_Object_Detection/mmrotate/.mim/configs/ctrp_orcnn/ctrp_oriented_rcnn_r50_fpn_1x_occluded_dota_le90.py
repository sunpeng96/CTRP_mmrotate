_base_ = [
    '../_base_/datasets/occluded_dotav1.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

work_dir = './work_dirs/ctrp_orcnn/srt_mask_oriented_rcnn_240502_2'

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
# S2REncoder design and Query components
with_SA = True
with_SA_pe = True
with_semantics_prior = True
with_spatial_prior = True
nheads = 6

if with_spatial_prior==False:
    find_unused_parameters = True

angle_version = 'le90'
model = dict(
    type='CTRPMaskOrientedRCNN',
    use_mask_img=mask_mode,  # 用于指示是否使用实例遮挡模块
    mask_ratio=mask_ratio,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CRTrMaskOrientedStandardRoIHead',
        with_mask_reason=mask_mode,  # with_mask_reason: 用于指示是否使用推理模块
        rel_pair_topK=rel_pair_topK,
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,  # with mask class
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        rel_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
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
                return_intermediate=True),
            loss_mask_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),  # 这个参数可能有用, 可以保证每个 batch中至少一个真值
            add_mask_gt_as_proposals=True,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            reason_nms=dict(iou_thr=0.1),  # 用于设置Mask_reason_head的 NMS阈值.
            max_per_img=2000)))

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))

evaluation = dict(interval=1, metric='mAP', eval_mode=eval_mode)  # 'ablation' or 'sota'

optimizer = dict(lr=0.0025*2*4)  # 0.0025 * workers_per_gpu * num_GPU
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './work_dirs/ctrp_orcnn/oriented_rcnn_r50_fpn_240501/latest.pth'
resume_from = None
work_dir = work_dir
workflow = [('train', 1)]
