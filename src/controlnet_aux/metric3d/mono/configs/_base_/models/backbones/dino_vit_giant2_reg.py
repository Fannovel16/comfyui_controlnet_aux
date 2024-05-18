model = dict(
    backbone=dict(
        type='vit_giant2_reg',
        prefix='backbones.',
        out_channels=[1536, 1536, 1536, 1536],
        drop_path_rate = 0.0),
    )
