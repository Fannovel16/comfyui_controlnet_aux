from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management

class AnimeLineArt_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(resolution=INPUT.RESOLUTION())

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, resolution=512, **kwargs):
        from custom_controlnet_aux.lineart_anime import LineartAnimeDetector

        model = LineartAnimeDetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution)
        del model
        return (out, )

NODE_CLASS_MAPPINGS = {
    "AnimeLineArtPreprocessor": AnimeLineArt_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimeLineArtPreprocessor": "Anime Lineart"
}