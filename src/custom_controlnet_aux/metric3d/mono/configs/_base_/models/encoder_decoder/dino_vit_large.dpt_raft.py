# model settings
_base_ = ['../backbones/dino_vit_large.py']
model = dict(
    type='DensePredModel',
    decode_head=dict(
        type='RAFTDepthDPT',
        in_channels=[1024, 1024, 1024, 1024],
        use_cls_token=True,
        feature_channels = [256, 512, 1024, 1024], # [2/7, 1/7, 1/14, 1/14]
        decoder_channels = [128, 256, 512, 1024, 1024], # [4/7, 2/7, 1/7, 1/14, 1/14]
        up_scale = 7,
        hidden_channels=[128, 128, 128, 128], # [x_4, x_8, x_16, x_32] [192, 384, 768, 1536]
        n_gru_layers=3,
        n_downsample=2,
        iters=12,
        slow_fast_gru=True,
        corr_radius=4,
        corr_levels=4,
        prefix='decode_heads.'),
)
