import warnings
import cv2
import numpy as np
from PIL import Image
from ..util import HWC3, resize_image

class BinaryDetector:
    def __call__(self, input_image=None, bin_threshold=0, detect_resolution=512, image_resolution=512, output_type=None, **kwargs):
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

        img_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        if bin_threshold == 0 or bin_threshold == 255:
        # Otsu's threshold
            otsu_threshold, detected_map = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            print("Otsu threshold:", otsu_threshold)
        else:
            _, img_bin = cv2.threshold(img_gray, bin_threshold, 255, cv2.THRESH_BINARY_INV)

        detected_map = HWC3(detected_map)      
         
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
