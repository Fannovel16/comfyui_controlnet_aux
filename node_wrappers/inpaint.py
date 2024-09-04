import torch
from ..utils import INPUT

class InpaintPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return dict(
            required=dict(image=INPUT.IMAGE(), mask=INPUT.MASK()),
            optional=dict(black_pixel_for_xinsir_cn=INPUT.BOOLEAN(False))
        )
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess"

    CATEGORY = "ControlNet Preprocessors/others"

    def preprocess(self, image, mask, black_pixel_for_xinsir_cn=False):
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(image.shape[1], image.shape[2]), mode="bilinear")
        mask = mask.movedim(1,-1).expand((-1,-1,-1,3))
        image = image.clone()
        if black_pixel_for_xinsir_cn:
            masked_pixel = 0.0
        else:
            masked_pixel = -1.0
        image[mask > 0.5] = masked_pixel
        return (image,)

NODE_CLASS_MAPPINGS = {
    "InpaintPreprocessor": InpaintPreprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintPreprocessor": "Inpaint Preprocessor"
}
