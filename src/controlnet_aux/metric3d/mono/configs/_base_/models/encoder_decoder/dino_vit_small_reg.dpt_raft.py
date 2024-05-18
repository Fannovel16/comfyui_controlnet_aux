# model settings
_base_ = ['../backbones/dino_vit_small_reg.py']
model = dict(
    type='DensePredModel',
    decode_head=dict(
        type='RAFTDepthDPT',
        in_channels=[384, 384, 384, 384],
        use_cls_token=True,
        feature_channels = [96, 192, 384, 768], # [2/7, 1/7, 1/14, 1/14]
        decoder_channels = [48, 96, 192, 384, 384], # [-, 1/4, 1/7, 1/14, 1/14]
        up_scale = 7,
        hidden_channels=[48, 48, 48, 48], # [x_4, x_8, x_16, x_32] [1/4, 1/7, 1/14, -]
        n_gru_layers=3,
        n_downsample=2,
        iters=3,
        slow_fast_gru=True,
        num_register_tokens=4,
        prefix='decode_heads.'),
)
