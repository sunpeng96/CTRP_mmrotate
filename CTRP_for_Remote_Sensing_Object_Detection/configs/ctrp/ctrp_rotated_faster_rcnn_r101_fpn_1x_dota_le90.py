_base_ = './ctrp_rotated_faster_rcnn_r50_fpn_1x_dota_le90.py'

learning_rate = 0.00125*2*4
work_dir = './work_dirs/dota/crtr_rotated_faster_rcnn_r101_fpn_1x_dota_le90_250417'

# model
model = dict(
    backbone=dict(depth=101,init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    train_cfg=dict(refined_rcnn=dict(sampler=dict(pos_fraction=0.75))),
    test_cfg=dict(
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000,
            # GauS
            # synth_cfg=dict(synth_thr=0.5, synth_method=2, alpha=1.0, beta=6.0)
        )))

# batchsize:2	lr:0.0025
optimizer = dict(lr=learning_rate)  # 0.0025 * workers_per_gpu * num_GPU
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
work_dir = work_dir
workflow = [('train', 1)]