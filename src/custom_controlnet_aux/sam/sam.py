"""
SAM implementation using HuggingFace transformers for PyTorch 2.7 compatibility.
"""
import numpy as np
import torch
from PIL import Image
from typing import Union

# Import utilities
from ..util import HWC3, common_input_validate, resize_image_with_pad


class SamDetector:
    
    def __init__(self, model_name="facebook/sam-vit-base"):
        from transformers import SamModel, SamProcessor
        
        self.model_name = model_name
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name)
        self.device = "cpu"

    @classmethod  
    def from_pretrained(cls, pretrained_model_or_path=None, model_type="vit_t", filename="mobile_sam.pt", subfolder=None):
        model_mapping = {
            "vit_t": "facebook/sam-vit-base",
            "vit_b": "facebook/sam-vit-base", 
            "vit_l": "facebook/sam-vit-large",
            "vit_h": "facebook/sam-vit-huge"
        }
        if filename and isinstance(filename, str):
            if "mobile_sam" in filename.lower():
                model_name = "facebook/sam-vit-base"
            elif "sam_vit_h" in filename.lower():
                model_name = "facebook/sam-vit-huge"
            elif "sam_vit_l" in filename.lower():
                model_name = "facebook/sam-vit-large"
            elif "sam_vit_b" in filename.lower():
                model_name = "facebook/sam-vit-base"
            else:
                model_name = model_mapping.get(model_type, "facebook/sam-vit-base")
        else:
            model_name = model_mapping.get(model_type, "facebook/sam-vit-base")
        
        return cls(model_name)

    def to(self, device):
        self.model = self.model.to(device) 
        self.device = device
        return self
        
    def generate_automatic_masks(self, input_image):
        if isinstance(input_image, np.ndarray):
            pil_image = Image.fromarray(input_image)
        else:
            pil_image = input_image
            
        height, width = pil_image.size[1], pil_image.size[0]
        
        points_per_side = max(8, min(24, width // 64, height // 64))
        
        grid_points = []
        for i in range(points_per_side):
            for j in range(points_per_side):
                x = int((j + 0.5) * width / points_per_side)
                y = int((i + 0.5) * height / points_per_side)
                x_offset = int((np.random.random() - 0.5) * (width / points_per_side * 0.3))
                y_offset = int((np.random.random() - 0.5) * (height / points_per_side * 0.3))
                x = max(5, min(width - 5, x + x_offset))
                y = max(5, min(height - 5, y + y_offset))
                grid_points.append([x, y])
        
        batch_size = 16
        all_masks = []
        
        for i in range(0, len(grid_points), batch_size):
            batch_points = grid_points[i:i + batch_size]
            input_points = [batch_points]
            
            inputs = self.processor(
                images=pil_image, 
                input_points=input_points, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            masks = self.processor.post_process_masks(
                outputs.pred_masks, 
                inputs["original_sizes"], 
                inputs["reshaped_input_sizes"]
            )[0]
            
            masks_np = masks.cpu().numpy()
            
            for j, mask in enumerate(masks_np):
                mask_2d = mask[0] if len(mask.shape) > 2 else mask
                area = int(mask_2d.sum())
                
                if area > 100:
                    cleaned_mask = self._postprocess_mask(mask_2d)
                    cleaned_area = int(cleaned_mask.sum())
                    
                    mask_dict = {
                        'segmentation': cleaned_mask,
                        'area': cleaned_area,
                        'stability_score': 0.88,
                        'point_coords': batch_points[j % len(batch_points)]
                    }
                    all_masks.append(mask_dict)
        
        return all_masks

    def _postprocess_mask(self, mask, min_region_area=100):
        from scipy import ndimage
        from skimage import morphology
        
        binary_mask = mask.astype(bool)
        
        filled_mask = ndimage.binary_fill_holes(binary_mask)
        
        if filled_mask.any():
            kernel_close = morphology.disk(5)
            kernel_open = morphology.disk(3)
            
            smoothed_mask = morphology.binary_closing(filled_mask, kernel_close)
            smoothed_mask = morphology.binary_opening(smoothed_mask, kernel_open)
            smoothed_mask = morphology.binary_closing(smoothed_mask, kernel_close)
        else:
            smoothed_mask = filled_mask
        
        return smoothed_mask.astype(mask.dtype)

    def show_anns(self, anns):
        if len(anns) == 0:
            return None
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        
        h, w = anns[0]['segmentation'].shape
        
        final_img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")
        
        for ann in sorted_anns:
            m = ann['segmentation']
            
            img = np.empty((m.shape[0], m.shape[1], 3), dtype=np.uint8)
            for i in range(3):
                img[:,:,i] = np.random.randint(255, dtype=np.uint8)
            
            final_img.paste(Image.fromarray(img, mode="RGB"), (0, 0), Image.fromarray(np.uint8(m*255)))
        
        return np.array(final_img, dtype=np.uint8)

    def __call__(self, input_image: Union[np.ndarray, Image.Image]=None, detect_resolution=512, output_type="pil", upscale_method="INTER_CUBIC", **kwargs) -> Image.Image:
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        masks = self.generate_automatic_masks(input_image)
        
        map = self.show_anns(masks)

        if map is None:
            map = np.zeros((input_image.shape[0], input_image.shape[1], 3), dtype=np.uint8)

        detected_map = HWC3(remove_pad(map))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map