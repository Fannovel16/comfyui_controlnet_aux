from ..utils import common_annotator_call, create_node_input_types, MAX_RESOLUTION
import comfy.model_management as model_management
import folder_paths

class Depth_Anything_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (["depth_anything_vitl14.pth", "depth_anything_vitb14.pth", "depth_anything_vits14.pth"], {"default": "depth_anything_vitl14.pth"}) }}
    RETURN_TYPES = ("DEPTH_MODEL",)
    FUNCTION = "load_checkpoint"

    CATEGORY = "ControlNet Preprocessors/Depth Loader"

    def load_checkpoint(self, ckpt_name):
        from controlnet_aux.depth_anything import DepthAnythingDetector
        model = DepthAnythingDetector.from_pretrained(filename=ckpt_name).to(model_management.get_torch_device())
        return (model, )


class Depth_Anything_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("DEPTH_MODEL",)
            },
            "optional": {
                "resolution": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, model, resolution=512, **kwargs):
        out = common_annotator_call(model, image, resolution=resolution)
        return (out, )

class Zoe_Depth_Anything_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "environment": (["indoor", "outdoor"], {"default": "indoor"})}}
    RETURN_TYPES = ("ZOEDEPTH_MODEL",)
    FUNCTION = "load_checkpoint"

    CATEGORY = "ControlNet Preprocessors/Depth Loader"

    def load_checkpoint(self, environment):
        from controlnet_aux.zoe import ZoeDepthAnythingDetector
        ckpt_name = "depth_anything_metric_depth_indoor.pt" if environment == "indoor" else "depth_anything_metric_depth_outdoor.pt"
        model = ZoeDepthAnythingDetector.from_pretrained(filename=ckpt_name).to(model_management.get_torch_device())
        return (model, )

class Zoe_Depth_Anything_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("ZOEDEPTH_MODEL",)
            },
            "optional": {
                "resolution": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, model, resolution=512, **kwargs):
        out = common_annotator_call(model, image, resolution=resolution)
        return (out, )

NODE_CLASS_MAPPINGS = {
    "DepthAnythingLoader": Depth_Anything_Loader,
    "DepthAnythingPreprocessor": Depth_Anything_Preprocessor,
    "Zoe_DepthAnythingLoader": Zoe_Depth_Anything_Loader,
    "Zoe_DepthAnythingPreprocessor": Zoe_Depth_Anything_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthAnythingLoader": "Depth Anything Loader",
    "DepthAnythingPreprocessor": "Depth Anything",
    "Zoe_DepthAnythingLoader": "Zoe Depth Anything Loader",
    "Zoe_DepthAnythingPreprocessor": "Zoe Depth Anything"
}