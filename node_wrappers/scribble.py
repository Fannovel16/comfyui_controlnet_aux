from ..utils import common_annotator_call, create_node_input_types, nms
import comfy.model_management as model_management
import cv2

class Scribble_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types()

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, resolution=512, **kwargs):
        from controlnet_aux.scribble import ScribbleDetector

        model = ScribbleDetector()
        return (common_annotator_call(model, image, resolution=resolution), )

class Scribble_XDoG_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            threshold = ("INT", {"default": 32, "min": 1, "max": 64, "step": 1})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, threshold, resolution=512, **kwargs):
        from controlnet_aux.scribble import ScribbleXDog_Detector

        model = ScribbleXDog_Detector()
        return (common_annotator_call(model, image, resolution=resolution, thr_a=threshold), )

class Scribble_PiDiNet_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            safe=(["enable", "disable"], {"default": "enable"})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, safe, resolution):
        def model(img, **kwargs):
            from controlnet_aux.pidi import PidiNetDetector
            pidinet = PidiNetDetector.from_pretrained().to(model_management.get_torch_device())
            result = pidinet(img, scribble=True, **kwargs)
            result = nms(result, 127, 3.0)
            result = cv2.GaussianBlur(result, (0, 0), 3.0)
            result[result > 4] = 255
            result[result < 255] = 0
            return result
        return (common_annotator_call(model, image, resolution=resolution, safe=safe=="enable"),)

NODE_CLASS_MAPPINGS = {
    "ScribblePreprocessor": Scribble_Preprocessor,
    "Scribble_XDoG_Preprocessor": Scribble_XDoG_Preprocessor,
    "Scribble_PiDiNet_Preprocessor": Scribble_PiDiNet_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ScribblePreprocessor": "Scribble Lines",
    "Scribble_XDoG_Preprocessor": "Scribble XDoG Lines",
    "Scribble_PiDiNet_Preprocessor": "Scribble PiDiNet Lines"
}
