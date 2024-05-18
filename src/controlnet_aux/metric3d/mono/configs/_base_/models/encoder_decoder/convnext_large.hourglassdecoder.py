# model settings
_base_ = ['../backbones/convnext_large.py',]
model = dict(
    type='DensePredModel',
    decode_head=dict(
        type='HourglassDecoder',
        in_channels=[192, 384, 768, 1536],
        decoder_channel=[128, 128, 256, 512],
        prefix='decode_heads.'),
)
