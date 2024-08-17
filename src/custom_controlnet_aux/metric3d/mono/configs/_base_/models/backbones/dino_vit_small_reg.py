model = dict(
    backbone=dict(
        type='vit_small_reg',
        prefix='backbones.',
        out_channels=[384, 384, 384, 384],
        drop_path_rate = 0.0),
    )
