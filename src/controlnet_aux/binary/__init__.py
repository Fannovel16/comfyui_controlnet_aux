import warnings
import cv2
import numpy as np
from PIL import Image
from controlnet_aux.util import HWC3, resize_image_with_pad

class BinaryDetector:
    def __call__(self, input_image=None, bin_threshold=0, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
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
        
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        img_gray = cv2.cvtColor(detected_map, cv2.COLOR_RGB2GRAY)
        if bin_threshold == 0 or bin_threshold == 255:
        # Otsu's threshold
            otsu_threshold, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            print("Otsu threshold:", otsu_threshold)
        else:
            _, img_bin = cv2.threshold(img_gray, bin_threshold, 255, cv2.THRESH_BINARY_INV)
        
        detected_map = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)
        detected_map = HWC3(remove_pad(255 - detected_map))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
