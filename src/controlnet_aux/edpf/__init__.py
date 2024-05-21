import warnings
import cv2
import numpy as np
from PIL import Image
from controlnet_aux.util import resize_image_with_pad, common_input_validate, HWC3

class EDPF:
    def __call__(self, input_image=None, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        ed = cv2.ximgproc.createEdgeDrawing()
        params = cv2.ximgproc.EdgeDrawing.Params()
        params.PFmode = True
        ed.setParams(params)
        edges = ed.detectEdges(cv2.cvtColor(detected_map, cv2.COLOR_BGR2GRAY))
        detected_map = ed.getEdgeImage(edges)
        detected_map = HWC3(remove_pad(detected_map))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
