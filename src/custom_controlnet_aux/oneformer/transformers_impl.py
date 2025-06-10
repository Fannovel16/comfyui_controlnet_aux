"""
Modern OneFormer implementation using transformers library
"""
import torch
import numpy as np
from PIL import Image
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from custom_controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad, HF_MODEL_NAME


MODEL_CONFIGS = {
    "coco": "shi-labs/oneformer_coco_swin_large",
    "ade20k": "shi-labs/oneformer_ade20k_swin_large"
}


class OneformerTransformersSegmentor:
    """Modern OneFormer implementation using transformers library"""
    
    def __init__(self, model_id):
        self.model_id = model_id
        self.processor = OneFormerProcessor.from_pretrained(model_id)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_id)
        self.device = "cpu"
        
    def to(self, device):
        """Move model to device"""
        self.device = device
        self.model = self.model.to(device)
        return self
    
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=HF_MODEL_NAME, filename="250_16_swin_l_oneformer_ade20k_160k.pth", config_path=None):
        """Create OneFormer model from pretrained weights"""
        if "coco" in filename.lower():
            dataset = "coco"
        elif "ade20k" in filename.lower():
            dataset = "ade20k"
        else:
            dataset = "ade20k"
        
        model_id = MODEL_CONFIGS[dataset]
        return cls(model_id)
    
    def __call__(self, input_image=None, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        """Process image for semantic segmentation"""
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        if isinstance(input_image, np.ndarray):
            pil_image = Image.fromarray(input_image)
        else:
            pil_image = input_image
            
        semantic_inputs = self.processor(
            images=pil_image, 
            task_inputs=["semantic"], 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**semantic_inputs)
        
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[pil_image.size[::-1]]
        )[0]
        
        seg_map = predicted_semantic_map.cpu().numpy().astype(np.uint8)
        detected_map = self._segmentation_to_colormap(seg_map)
        detected_map = remove_pad(HWC3(detected_map))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
    
    def _segmentation_to_colormap(self, seg_map):
        """Convert segmentation map to colored visualization"""
        height, width = seg_map.shape
        color_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        max_classes = seg_map.max() + 1
        colors = np.random.RandomState(42).randint(0, 255, (max_classes, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]
        
        for class_id in range(max_classes):
            mask = seg_map == class_id
            color_map[mask] = colors[class_id]
            
        return color_map


OneformerSegmentor = OneformerTransformersSegmentor