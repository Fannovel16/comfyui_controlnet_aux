import warnings
import cv2
import numpy as np
from PIL import Image
from custom_controlnet_aux.util import HWC3, resize_image_with_pad, common_input_validate, HWC3

#Not to be confused with "scribble" from HED. That is "fake scribble" which is more accurate and less picky than this.
class ScribbleDetector:
    def __call__(self, input_image=None, detect_resolution=512, output_type=None, upscale_method="INTER_AREA", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        detected_map = np.zeros_like(input_image, dtype=np.uint8)
        detected_map[np.min(input_image, axis=2) < 127] = 255
        detected_map = 255 - detected_map

        detected_map = remove_pad(detected_map)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map

class ScribbleXDog_Detector:
    def __call__(self, input_image=None, detect_resolution=512, thr_a=32, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        g1 = cv2.GaussianBlur(input_image.astype(np.float32), (0, 0), 0.5)
        g2 = cv2.GaussianBlur(input_image.astype(np.float32), (0, 0), 5.0)
        dog = (255 - np.min(g2 - g1, axis=2)).clip(0, 255).astype(np.uint8)
        result = np.zeros_like(input_image, dtype=np.uint8)
        result[2 * (255 - dog) > thr_a] = 255
        #result = 255 - result

        detected_map = HWC3(remove_pad(result))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map