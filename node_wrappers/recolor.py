from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT

class ImageLuminanceDetector:
    @classmethod
    def INPUT_TYPES(s):
        #https://github.com/Mikubill/sd-webui-controlnet/blob/416c345072c9c2066101e225964e3986abe6945e/scripts/processor.py#L1229
        return define_preprocessor_inputs(
            gamma_correction=INPUT.FLOAT(default=1.0, min=0.1, max=2.0),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Recolor"

    def execute(self, image, gamma_correction=1.0, resolution=512, **kwargs):
        from custom_controlnet_aux.recolor import Recolorizer
        return (common_annotator_call(Recolorizer(), image, mode="luminance", gamma_correction=gamma_correction , resolution=resolution), )

class ImageIntensityDetector:
    @classmethod
    def INPUT_TYPES(s):
        #https://github.com/Mikubill/sd-webui-controlnet/blob/416c345072c9c2066101e225964e3986abe6945e/scripts/processor.py#L1229
        return define_preprocessor_inputs(
            gamma_correction=INPUT.FLOAT(default=1.0, min=0.1, max=2.0),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Recolor"

    def execute(self, image, gamma_correction=1.0, resolution=512, **kwargs):
        from custom_controlnet_aux.recolor import Recolorizer
        return (common_annotator_call(Recolorizer(), image, mode="intensity", gamma_correction=gamma_correction , resolution=resolution), )

NODE_CLASS_MAPPINGS = {
    "ImageLuminanceDetector": ImageLuminanceDetector,
    "ImageIntensityDetector": ImageIntensityDetector
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageLuminanceDetector": "Image Luminance",
    "ImageIntensityDetector": "Image Intensity"
}