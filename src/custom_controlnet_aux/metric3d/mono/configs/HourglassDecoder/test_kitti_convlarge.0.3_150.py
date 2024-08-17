_base_=[
       '../_base_/models/encoder_decoder/convnext_large.hourglassdecoder.py',
       '../_base_/datasets/_data_base_.py',
       '../_base_/default_runtime.py',
       ]

model = dict(
    backbone=dict(
        pretrained=False,
    )
)

# configs of the canonical space
data_basic=dict(
    canonical_space = dict(
        img_size=(512, 960),
        focal_length=1000.0,
    ),
    depth_range=(0, 1),
    depth_normalize=(0.3, 150),
    crop_size = (512, 1088),
) 

batchsize_per_gpu = 2
thread_per_gpu = 4
