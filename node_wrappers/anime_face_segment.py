from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management
import torch
from einops import rearrange

class AnimeFace_SemSegPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        #This preprocessor is only trained on 512x resolution
        #https://github.com/siyeong0/Anime-Face-Segmentation/blob/main/predict.py#L25
        return define_preprocessor_inputs(
            remove_background_using_abg=INPUT.BOOLEAN(True),
            resolution=INPUT.RESOLUTION(default=512, min=512, max=512)
        )

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "ABG_CHARACTER_MASK (MASK)")
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Semantic Segmentation"

    def execute(self, image, remove_background_using_abg=True, resolution=512, **kwargs):
        from custom_controlnet_aux.anime_face_segment import AnimeFaceSegmentor

        model = AnimeFaceSegmentor.from_pretrained().to(model_management.get_torch_device())
        if remove_background_using_abg:
            out_image_with_mask = common_annotator_call(model, image, resolution=resolution, remove_background=True)
            out_image = out_image_with_mask[..., :3]
            mask = out_image_with_mask[..., 3:]
            mask = rearrange(mask, "n h w c -> n c h w")
        else:
            out_image = common_annotator_call(model, image, resolution=resolution, remove_background=False)
            N, H, W, C = out_image.shape
            mask = torch.ones(N, C, H, W)
        del model
        return (out_image, mask)

NODE_CLASS_MAPPINGS = {
    "AnimeFace_SemSegPreprocessor": AnimeFace_SemSegPreprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimeFace_SemSegPreprocessor": "Anime Face Segmentor"
}