import cv2
import warnings
import cv2
import numpy as np
from PIL import Image
from controlnet_aux.util import HWC3, safer_memory, common_input_validate

def cv2_resize_shortest_edge(image, size):
    h, w = image.shape[:2]
    if h < w:
        new_h = size
        new_w = int(round(w / h * size))
    else:
        new_w = size
        new_h = int(round(h / w * size))
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image

def apply_color(img, res=512):
    img = cv2_resize_shortest_edge(img, res)
    h, w = img.shape[:2]

    input_img_color = cv2.resize(img, (w//64, h//64), interpolation=cv2.INTER_CUBIC)  
    input_img_color = cv2.resize(input_img_color, (w, h), interpolation=cv2.INTER_NEAREST)
    return input_img_color

#Color T2I like multiples-of-64, upscale methods are fixed.
class ColorDetector:
    def __call__(self, input_image=None, detect_resolution=512, output_type=None, **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image = HWC3(input_image)
        detected_map = HWC3(apply_color(input_image, detect_resolution))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
