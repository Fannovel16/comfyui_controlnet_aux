import warnings
import cv2
import numpy as np
from PIL import Image
from ..util import HWC3, resize_image

#Not to be confused with "scribble" from HED. That is "fake scribble" which is more accurate and less picky than this.
class PickyScribble:
    def __call__(self, input_image=None, detect_resolution=512, image_resolution=512, output_type=None, **kwargs):
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

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255    
         
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map