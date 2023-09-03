import warnings
import cv2
import numpy as np
from PIL import Image
from ..util import HWC3, resize_image


class TileDetector:
    def __call__(self, input_image=None, pyrUp_iters=3, detect_resolution=512, image_resolution=512, output_type=None,
                 **kwargs):
        if "img" in kwargs:
            warnings.warn("img is deprecated, please use `input_image=...` instead.", DeprecationWarning)
            input_image = kwargs.pop("img")

        if input_image is None:
            raise ValueError("input_image must be defined.")

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
            output_type = output_type or "pil"
        else:
            output_type = output_type or "np"

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        detected_map = HWC3(input_image)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W // (2 ** pyrUp_iters), H // (2 ** pyrUp_iters)),
                                  interpolation=cv2.INTER_AREA)
        for _ in range(pyrUp_iters):
            detected_map = cv2.pyrUp(detected_map)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
