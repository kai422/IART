# CUDA_VISIBLE_DEVICES=4,5 PORT=12345 ./tools/dist_train.sh configs/restorers/sintel_swin/swin_vsr_sintel_feature_alignment_nearest.py 2
 
exp_name = 'swin_vsr_sintel_feature_alignment_nearest'

# model settings
model = dict(
    type='VSRFlowInput',
    generator=dict(
        type='SintelSwinVSRNetFeatureAlignment', alignment = 'of_warp', interpolation = 'nearest'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
train_dataset_type = 'SRSintelMultipleGTDataset'
val_dataset_type = 'SRSintelMultipleGTDataset'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1], filename_tmpl='frame_{:04d}.png', start_idx=1),
    dict(type='TemporalReverse', keys='lq_path', reverse_ratio=0),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(
        type='LoadSintelOpticalFlow',
        io_backend='disk'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='LqGtFlowRandomCrop', gt_patch_size=256),
    dict(type='LqGtFlowFlip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='LqGtFlowFlip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='LqGtFlowRandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt', 'flows_forward']),
    dict(type='Collect', keys=['lq','flows_forward', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1], filename_tmpl='frame_{:04d}.png', start_idx=1),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(
        type='LoadSintelOpticalFlow',
        io_backend='disk'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt', 'flows_forward']),
    dict(
        type='Collect',
        keys=['lq','flows_forward', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1], filename_tmpl='frame_{:04d}.png', start_idx=1),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadSintelOpticalFlow',
        io_backend='disk'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq', 'flows_forward']),
    dict(type='Collect', keys=['lq','flows_forward'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),  # 2 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/sintel/training/clean_bicubic_X4',
            gt_folder='data/sintel/training/clean',
            num_input_frames=6,
            pipeline=train_pipeline,
            scale=4,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='data/sintel/training/clean_bicubic_X4',
        gt_folder='data/sintel/training/clean',
        num_input_frames=None,
        pipeline=test_pipeline,
        scale=4,
        repeat=2,
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
        lq_folder='data/sintel/training/clean_bicubic_X4',
        gt_folder='data/sintel/training/clean',
        num_input_frames=None,
        pipeline=test_pipeline,
        scale=4,
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=2e-4,
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)})))

# learning policy
total_iters = 100000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[100000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
