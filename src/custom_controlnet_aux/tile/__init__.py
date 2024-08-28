import warnings
import cv2
import numpy as np
from PIL import Image
from custom_controlnet_aux.util import get_upscale_method, common_input_validate, HWC3
from .guided_filter import FastGuidedFilter

class TileDetector:
    def __call__(self, input_image=None, pyrUp_iters=3, output_type=None, upscale_method="INTER_AREA", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        H, W, _ = input_image.shape
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
        detected_map = cv2.resize(input_image, (W // (2 ** pyrUp_iters), H // (2 ** pyrUp_iters)),
                                  interpolation=get_upscale_method(upscale_method))
        detected_map = HWC3(detected_map)

        for _ in range(pyrUp_iters):
            detected_map = cv2.pyrUp(detected_map)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map


# Source: https://huggingface.co/TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic/blob/main/TTP_tile_preprocessor_v5.py

def apply_gaussian_blur(image_np, ksize=5, sigmaX=1.0):
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=sigmaX)
    return blurred_image

def apply_guided_filter(image_np, radius, eps, scale):
    filter = FastGuidedFilter(image_np, radius, eps, scale)
    return filter.filter(image_np)

class TTPlanet_Tile_Detector_GF:
    def __call__(self, input_image, scale_factor, blur_strength, radius, eps, output_type=None, **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        img_np = input_image[:, :, ::-1] # RGB to BGR
        
        # Apply Gaussian blur
        img_np = apply_gaussian_blur(img_np, ksize=int(blur_strength), sigmaX=blur_strength / 2)            

        # Apply Guided Filter
        img_np = apply_guided_filter(img_np, radius, eps, scale_factor)

        # Resize image
        height, width = img_np.shape[:2]
        new_width = int(width / scale_factor)
        new_height = int(height / scale_factor)
        resized_down = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_img = cv2.resize(resized_down, (width, height), interpolation=cv2.INTER_CUBIC)
        detected_map = HWC3(resized_img[:, :, ::-1]) # BGR to RGB
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        
        return detected_map

class TTPLanet_Tile_Detector_Simple:
    def __call__(self, input_image, scale_factor, blur_strength, output_type=None, **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        img_np = input_image[:, :, ::-1] # RGB to BGR
        
        # Resize image first if you want blur to apply after resizing
        height, width = img_np.shape[:2]
        new_width = int(width / scale_factor)
        new_height = int(height / scale_factor)
        resized_down = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_img = cv2.resize(resized_down, (width, height), interpolation=cv2.INTER_LANCZOS4)
    
        # Apply Gaussian blur after resizing
        img_np = apply_gaussian_blur(resized_img, ksize=int(blur_strength), sigmaX=blur_strength / 2)
        detected_map = HWC3(img_np[:, :, ::-1]) # BGR to RGB
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        
        return detected_map
