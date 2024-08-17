import warnings
from typing import Union

import cv2
import numpy as np
from PIL import Image

from custom_controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad
from .mediapipe_face_common import generate_annotation


class MediapipeFaceDetector:
    def __call__(self,
                 input_image: Union[np.ndarray, Image.Image] = None,
                 max_faces: int = 1,
                 min_confidence: float = 0.5,
                 output_type: str = "pil",
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 upscale_method="INTER_CUBIC",
                 **kwargs):

        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        detected_map = generate_annotation(detected_map, max_faces, min_confidence)
        detected_map = remove_pad(HWC3(detected_map))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
