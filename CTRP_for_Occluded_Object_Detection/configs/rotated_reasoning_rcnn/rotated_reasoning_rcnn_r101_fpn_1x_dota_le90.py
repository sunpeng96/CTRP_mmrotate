_base_ = './rotated_reasoning_rcnn_r50_fpn_1x_dota_le90.py'

work_dir = 'work_dirs/rotated_reasoning_rcnn_20240831'

# model
model = dict(backbone=dict(depth=101,init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')))

# batchsize:2	lr:0.0025
optimizer = dict(lr=0.00125*2*4)  # 0.0025 * workers_per_gpu * num_GPU
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
work_dir = work_dir
workflow = [('train', 1)]