import warnings
import cv2
import numpy as np
from PIL import Image
from controlnet_aux.util import get_upscale_method, common_input_validate, HWC3


class TileDetector:
    def __call__(self, input_image=None, pyrUp_iters=3, output_type=None, upscale_method="INTER_AREA", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        H, W, _ = input_image.shape
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
        detected_map = cv2.resize(input_image, (W // (2 ** pyrUp_iters), H // (2 ** pyrUp_iters)),
                                  interpolation=get_upscale_method(upscale_method))
        detected_map = HWC3(detected_map)

        for _ in range(pyrUp_iters):
            detected_map = cv2.pyrUp(detected_map)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
