# model settings
_base_ = ['../backbones/dino_vit_giant2_reg.py']
model = dict(
    type='DensePredModel',
    decode_head=dict(
        type='RAFTDepthDPT',
        in_channels=[1536, 1536, 1536, 1536],
        use_cls_token=True,
        feature_channels = [384, 768, 1536, 1536], # [2/7, 1/7, 1/14, 1/14]
        decoder_channels = [192, 384, 768, 1536, 1536], # [4/7, 2/7, 1/7, 1/14, 1/14]
        up_scale = 7,
        hidden_channels=[192, 192, 192, 192], # [x_4, x_8, x_16, x_32] [192, 384, 768, 1536]
        n_gru_layers=3,
        n_downsample=2,
        iters=3,
        slow_fast_gru=True,
        num_register_tokens=4,
        prefix='decode_heads.'),
)
