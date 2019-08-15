# model settings
norm_cfg = dict(type='BN', requires_grad=False)
model = dict(
    type='EncoderDecoder',
    pretrained='modelzoo://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        # strides=(1, 2, 2),
        # dilations=(1, 1, 1),
        out_indices=(3, ),
        frozen_stages=3,
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='pytorch'),
    shared_head=None,
    rpn_head=None,
    bbox_head=None,
    seg_cfg=None,
    cap_cfg=dict(
        encoder_dim=2048,
        embed_dim=512,
        attention_dim=512,
        decoder_dim=512,
        vocab_size=9490))
    # shared_head=dict(
    #     type='ResLayer',
    #     depth=50,
    #     stage=3,
    #     stride=2,
    #     dilation=1,
    #     style='caffe',
    #     norm_cfg=norm_cfg,
    #     norm_eval=True),
    # rpn_head=dict(
    #     type='RPNHead',
    #     in_channels=1024,
    #     feat_channels=1024,
    #     anchor_scales=[2, 4, 8, 16, 32],
    #     anchor_ratios=[0.5, 1.0, 2.0],
    #     anchor_strides=[16],
    #     target_means=[.0, .0, .0, .0],
    #     target_stds=[1.0, 1.0, 1.0, 1.0],
    #     loss_cls=dict(
    #         type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    #     loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    # bbox_roi_extractor=dict(
    #     type='SingleRoIExtractor',
    #     roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
    #     out_channels=1024,
    #     featmap_strides=[16]),
    # bbox_head=dict(
    #     type='BBoxHead',
    #     with_avg_pool=True,
    #     roi_feat_size=7,
    #     in_channels=2048,
    #     num_classes=81,
    #     target_means=[0., 0., 0., 0.],
    #     target_stds=[0.1, 0.1, 0.2, 0.2],
    #     reg_class_agnostic=False,
    #     loss_cls=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    #     loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    # rpn=None,
    # rpn_proposal=None,
    # rcnn=None)
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
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
        nms_across_levels=False,
        nms_pre=12000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=6000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100))
# dataset settings
dataset_type = 'MyCocoDataset'
data_root = '/mnt/coco17/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[58.395, 57.12, 57.375], to_rgb=False)
data = dict(
    imgs_per_gpu=20,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        img_scale=(256, 256),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True,
        with_semantic_seg=False,
        with_cap=True,
        split='Train',
        cap_f=data_root+'annotations/caps_coco17.json',
        cap_dir=data_root+'annotations/caps/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(256, 256),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(256, 256),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='Adam', lr=4e-4)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=1))
# learning policy
lr_config = dict(
    policy='fixed',
    )
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/mnt/work_dirs/cap_r101_adam_256_stage1'
load_from = None
resume_from = None
workflow = [('train', 1)]
