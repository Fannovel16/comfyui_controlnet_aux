from ..utils import common_annotator_call, create_node_input_types


class Tile_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            pyrUp_iters = ("INT", {"default": 3, "min": 1, "max": 10, "step": 1})
        )
        

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/tile"

    def execute(self, image, pyrUp_iters, resolution=512, **kwargs):
        from controlnet_aux.tile import TileDetector

        return (common_annotator_call(TileDetector(), image, pyrUp_iters=pyrUp_iters, resolution=resolution),)

class TTPlanet_TileGF_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.00, "min": 1.00, "max": 8.00, "step": 0.05}),
                "blur_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "radius": ("INT", {"default": 7, "min": 1, "max": 20, "step": 1}),
                "eps": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/tile"

    def execute(self, image, scale_factor, blur_strength, radius, eps, **kwargs):
        from controlnet_aux.tile import TTPlanet_Tile_Detector_GF

        return (common_annotator_call(TTPlanet_Tile_Detector_GF(), image, scale_factor=scale_factor, blur_strength=blur_strength, radius=radius, eps=eps),)

class TTPlanet_TileSimple_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.00, "min": 1.00, "max": 8.00, "step": 0.05}),
                "blur_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/tile"

    def execute(self, image, scale_factor, blur_strength):
        from controlnet_aux.tile import TTPLanet_Tile_Detector_Simple

        return (common_annotator_call(TTPLanet_Tile_Detector_Simple(), image, scale_factor=scale_factor, blur_strength=blur_strength),)


NODE_CLASS_MAPPINGS = {
    "TilePreprocessor": Tile_Preprocessor,
    "TTPlanet_TileGF_Preprocessor": TTPlanet_TileGF_Preprocessor,
    "TTPlanet_TileSimple_Preprocessor": TTPlanet_TileSimple_Preprocessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TilePreprocessor": "Tile",
    "TTPlanet_TileGF_Preprocessor": "TTPlanet Tile GuidedFilter",
    "TTPlanet_TileSimple_Preprocessor": "TTPlanet Tile Simple"
}
