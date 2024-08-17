import warnings
import cv2
import numpy as np
from PIL import Image
from custom_controlnet_aux.util import resize_image_with_pad, common_input_validate, HWC3

class CannyDetector:
    def __call__(self, input_image=None, low_threshold=100, high_threshold=200, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        detected_map = cv2.Canny(detected_map, low_threshold, high_threshold)
        detected_map = HWC3(remove_pad(detected_map))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
