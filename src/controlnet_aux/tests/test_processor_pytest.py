import io

import numpy as np
import pytest
from PIL import Image

from controlnet_aux.processor import MODELS, Processor


@pytest.fixture(params=[
    'scribble_hed',
    'softedge_hed',
    'scribble_hedsafe',
    'softedge_hedsafe',
    'depth_midas',
    'mlsd',
    'openpose',
    'openpose_hand',
    'openpose_face',
    'openpose_faceonly',
    'openpose_full',
    'scribble_pidinet',
    'softedge_pidinet',
    'scribble_pidsafe',
    'softedge_pidsafe',
    'normal_bae',
    'lineart_coarse',
    'lineart_realistic',
    'lineart_anime',
    'canny',
    'shuffle',
    'depth_zoe',
    'depth_leres',
    'depth_leres++',
    'mediapipe_face',
    'tile'
])
def processor(request):
    return Processor(request.param)


def test_processor_init(processor):
    assert isinstance(processor.processor, MODELS[processor.processor_id]['class'])
    assert isinstance(processor.params, dict)


def test_processor_call(processor):
    # Load test image
    with open('test_image.png', 'rb') as f:
        image_bytes = f.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Output size
    resolution = 512
    W, H = image.size
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64

    # Test processing
    processed_image = processor(image)
    assert isinstance(processed_image, Image.Image)
    assert processed_image.size == (W, H)


def test_processor_call_bytes(processor):
    # Load test image
    with open('test_image.png', 'rb') as f:
        image_bytes = f.read()

    # Test processing
    processed_image_bytes = processor(image_bytes, to_pil=False)
    assert isinstance(processed_image_bytes, bytes)
    assert len(processed_image_bytes) > 0