"""
OneFormer implementation using HuggingFace transformers for PyTorch 2.7 compatibility.
Provides equivalent functionality to the original detectron2 implementation.
"""
import numpy as np
import cv2
import torch
from PIL import Image

# Import utilities
from ..util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, HF_MODEL_NAME


class OneformerSegmentor:
    """
    OneFormer segmentation using HuggingFace transformers implementation.
    
    Uses equivalent models that are PyTorch 2.7 compatible and actively maintained:
    - Same architecture (OneFormer with Swin-Large backbone)
    - Same training datasets (COCO panoptic / ADE20K)
    - Professional colorized visualization output
    """
    
    def __init__(self, model_name):
        """Initialize OneFormer with HuggingFace transformers implementation."""
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        
        self.model_name = model_name
        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
        self.device = "cpu"

    @classmethod  
    def from_pretrained(cls, pretrained_model_or_path=HF_MODEL_NAME, filename="250_16_swin_l_oneformer_ade20k_160k.pth", config_path=None):
        """Create OneFormer model from pretrained weights."""
        model_mapping = {
            "250_16_swin_l_oneformer_ade20k_160k.pth": "shi-labs/oneformer_ade20k_swin_large",
            "150_16_swin_l_oneformer_coco_100ep.pth": "shi-labs/oneformer_coco_swin_large"
        }
        
        if filename in model_mapping:
            model_name = model_mapping[filename]
        elif "coco" in filename.lower():
            model_name = "shi-labs/oneformer_coco_swin_large"
        else:
            model_name = "shi-labs/oneformer_ade20k_swin_large"
        
        return cls(model_name)

    def to(self, device):
        """Move model to specified device."""
        self.model = self.model.to(device) 
        self.device = device
        return self
        
    def __call__(self, input_image=None, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        """Process image for semantic segmentation."""
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        # Convert to PIL for processing
        if isinstance(input_image, np.ndarray):
            pil_image = Image.fromarray(input_image)
        else:
            pil_image = input_image
            
        # Process with HuggingFace pipeline
        semantic_inputs = self.processor(
            images=pil_image, 
            task_inputs=["semantic"], 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**semantic_inputs)
        
        # Post-process results
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[pil_image.size[::-1]]
        )[0]
        
        # Convert to colormap using professional color scheme
        seg_map = predicted_semantic_map.cpu().numpy().astype(np.uint8)
        detected_map = self._generate_professional_colormap(seg_map)
        detected_map = remove_pad(HWC3(detected_map))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
    
    def _generate_professional_colormap(self, seg_map):
        """Generate professional colormap for segmentation visualization."""
        height, width = seg_map.shape
        color_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        max_possible_classes = 200
        colors = self._generate_detectron2_style_palette(max_possible_classes)
        
        unique_classes = np.unique(seg_map)
        for class_id in unique_classes:
            mask = seg_map == class_id
            color_map[mask] = colors[class_id % len(colors)]
            
        return color_map
    
    def _generate_detectron2_style_palette(self, num_classes):
        """Generate professional color palette with good visual separation."""
        colors = np.zeros((num_classes, 3), dtype=np.uint8)
        
        colors[0] = [0, 0, 0]  # Background is black
        
        for i in range(1, num_classes):
            hue = (i * 137.508) % 360  # Golden angle for good distribution
            saturation = 0.6 + 0.4 * ((i % 3) / 2)  # 60-100% saturation
            value = 0.7 + 0.3 * ((i % 2))  # 70-100% value
            
            color_hsv = np.array([[[hue / 2, saturation * 255, value * 255]]], dtype=np.uint8)
            color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0]
            colors[i] = color_rgb
            
        return colors


