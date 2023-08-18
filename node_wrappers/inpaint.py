import torch

def preprocess(image, mask):
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(image.shape[1], image.shape[2]), mode="bilinear")
    mask = mask.movedim(1,-1).expand((-1,-1,-1,3))
    image = image.clone()
    image[mask > 0.5] = -1.0  # set as masked pixel
    return image

class InpaintPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "mask": ("MASK",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess"

    CATEGORY = "ControlNet Preprocessors/others"

    def preprocess(self, image, mask):
        return (preprocess(image, mask),)

NODE_CLASS_MAPPINGS = {
    "InpaintPreprocessor": InpaintPreprocessor
}
NODE_CLASS_DISPLAY_MAPPINGS = {
    "InpaintPreprocessor": "Inpaint Preprocessor"
}