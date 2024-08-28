_base_=[
       '../_base_/models/encoder_decoder/dino_vit_small_reg.dpt_raft.py',
       '../_base_/datasets/_data_base_.py',
       '../_base_/default_runtime.py',
       ]

model=dict(
    decode_head=dict(
        type='RAFTDepthNormalDPT5',
        iters=4,
        n_downsample=2,
        detach=False,
    )
)


max_value = 200
# configs of the canonical space
data_basic=dict(
    canonical_space = dict(
        # img_size=(540, 960),
        focal_length=1000.0,
    ),
    depth_range=(0, 1),
    depth_normalize=(0.1, max_value),
    crop_size = (616, 1064),  # %28 = 0
     clip_depth_range=(0.1, 200),
    vit_size=(616,1064)
) 

batchsize_per_gpu = 1
thread_per_gpu = 1
