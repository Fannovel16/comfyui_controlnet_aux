import warnings
import cv2
import numpy as np
from PIL import Image
from custom_controlnet_aux.util import resize_image_with_pad, common_input_validate, HWC3

#https://github.com/Mikubill/sd-webui-controlnet/blob/416c345072c9c2066101e225964e3986abe6945e/scripts/processor.py#L639
def recolor_luminance(img, thr_a=1.0):
    result = cv2.cvtColor(HWC3(img), cv2.COLOR_BGR2LAB)
    result = result[:, :, 0].astype(np.float32) / 255.0
    result = result ** thr_a
    result = (result * 255.0).clip(0, 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def recolor_intensity(img, thr_a=1.0):
    result = cv2.cvtColor(HWC3(img), cv2.COLOR_BGR2HSV)
    result = result[:, :, 2].astype(np.float32) / 255.0
    result = result ** thr_a
    result = (result * 255.0).clip(0, 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result

recolor_methods = {
    "luminance": recolor_luminance,
    "intensity": recolor_intensity
}

class Recolorizer:
    def __call__(self, input_image=None, mode="luminance", gamma_correction=1.0, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        assert mode in recolor_methods.keys()
        detected_map = recolor_methods[mode](input_image, gamma_correction)
        detected_map = HWC3(remove_pad(detected_map))
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        return detected_map
