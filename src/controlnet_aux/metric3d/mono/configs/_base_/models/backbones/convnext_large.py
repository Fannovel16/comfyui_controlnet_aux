#_base_ = ['./_model_base_.py',]

#'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth' 
model = dict(
    #type='EncoderDecoderAuxi',
    backbone=dict(
        type='convnext_large',
        pretrained=True, 
        in_22k=True,
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        checkpoint='data/pretrained_weight_repo/convnext/convnext_large_22k_1k_384.pth',
        prefix='backbones.',
        out_channels=[192, 384, 768, 1536]),
    )
