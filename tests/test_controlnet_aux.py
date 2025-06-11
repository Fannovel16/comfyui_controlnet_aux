import os
import shutil
from io import BytesIO

import numpy as np
import pytest
import requests
from PIL import Image

from custom_controlnet_aux import (CannyDetector, ContentShuffleDetector, HEDdetector,
                            LeresDetector, LineartAnimeDetector,
                            LineartDetector, MediapipeFaceDetector,
                            MidasDetector, MLSDdetector, NormalBaeDetector,
                            OpenposeDetector, PidiNetDetector, SamDetector,
                            ZoeDetector, TileDetector)

OUTPUT_DIR = "tests/outputs"

def output(name, img):
    img.save(os.path.join(OUTPUT_DIR, "{:s}.png".format(name)))

def common(name, processor, img):
    output(name, processor(img))
    output(name + "_pil_np", Image.fromarray(processor(img, output_type="np")))
    output(name + "_np_np", Image.fromarray(processor(np.array(img, dtype=np.uint8), output_type="np")))
    output(name + "_np_pil", processor(np.array(img, dtype=np.uint8), output_type="pil"))
    output(name + "_scaled", processor(img, detect_resolution=640, image_resolution=768))

def return_pil(name, processor, img):
    output(name + "_pil_false", Image.fromarray(processor(img, return_pil=False)))
    output(name + "_pil_true", processor(img, return_pil=True))

@pytest.fixture(scope="module")
def img():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)
    url = "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))
    return img

def test_canny(img):
    canny = CannyDetector()
    common("canny", canny, img)
    output("canny_img", canny(img=img))

def test_hed(img):
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    common("hed", hed, img)
    return_pil("hed", hed, img)
    output("hed_safe", hed(img, safe=True))
    output("hed_scribble", hed(img, scribble=True))

def test_leres(img):
    leres = LeresDetector.from_pretrained("lllyasviel/Annotators")
    common("leres", leres, img)
    output("leres_boost", leres(img, boost=True))

def test_lineart(img):
    lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
    common("lineart", lineart, img)
    return_pil("lineart", lineart, img)
    output("lineart_coarse", lineart(img, coarse=True))

def test_lineart_anime(img):
    lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    common("lineart_anime", lineart_anime, img)
    return_pil("lineart_anime", lineart_anime, img)

def test_mediapipe_face(img):
    mediapipe = MediapipeFaceDetector()
    common("mediapipe", mediapipe, img)
    output("mediapipe_image", mediapipe(image=img))

def test_midas(img):
    midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
    common("midas", midas, img)
    output("midas_normal", midas(img, depth_and_normal=True)[1])

def test_mlsd(img):
    mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators")
    common("mlsd", mlsd, img)
    return_pil("mlsd", mlsd, img)

def test_normalbae(img):
    normal_bae = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
    common("normal_bae", normal_bae, img)
    return_pil("normal_bae", normal_bae, img)

def test_openpose(img):
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    common("openpose", openpose, img)
    return_pil("openpose", openpose, img)
    output("openpose_hand_and_face_false", openpose(img, hand_and_face=False))
    output("openpose_hand_and_face_true", openpose(img, hand_and_face=True))
    output("openpose_face", openpose(img, include_body=True, include_hand=False, include_face=True))
    output("openpose_faceonly", openpose(img, include_body=False, include_hand=False, include_face=True))
    output("openpose_full", openpose(img, include_body=True, include_hand=True, include_face=True))
    output("openpose_hand", openpose(img, include_body=True, include_hand=True, include_face=False))

def test_pidi(img):
    pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
    common("pidi", pidi, img)
    return_pil("pidi", pidi, img)
    output("pidi_safe", pidi(img, safe=True))
    output("pidi_scribble", pidi(img, scribble=True))

def test_sam(img):
    sam = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
    common("sam", sam, img)
    output("sam_image", sam(image=img))

def test_shuffle(img):
    shuffle = ContentShuffleDetector()
    common("shuffle", shuffle, img)
    return_pil("shuffle", shuffle, img)

def test_zoe(img):
    zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
    common("zoe", zoe, img)

def test_tile(img):
    tile = TileDetector()
    common("tile", tile, img)
    output("tile_img", tile(img))