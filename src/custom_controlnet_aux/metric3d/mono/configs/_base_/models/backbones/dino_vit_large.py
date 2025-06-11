model = dict(
    backbone=dict(
        type='vit_large',
        prefix='backbones.',
        out_channels=[1024, 1024, 1024, 1024],
        drop_path_rate = 0.0),
    )
