from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME, DWPOSE_MODEL_NAME, create_node_input_types
import comfy.model_management as model_management
import numpy as np
import warnings
from controlnet_aux.dwpose import DwposeDetector
import os

#Trigger startup caching for onnxruntime
ONNX_PROVIDERS = ["CUDAExecutionProvider", "DirectMLExecutionProvider", "OpenVINOExecutionProvider", "ROCMExecutionProvider"]
def check_ort_gpu():
    try:
        import onnxruntime as ort
        for provider in ONNX_PROVIDERS:
            if provider in ort.get_available_providers():
                return True
        return False
    except:
        return False

if not os.environ.get("DWPOSE_ONNXRT_CHECKED"):
    if check_ort_gpu():
        print("DWPose: Onnxruntime with acceleration providers detected")
    else:
        warnings.warn("DWPose: Onnxruntime not found or doesn't come with acceleration providers, switch to OpenCV with CPU device. DWPose might run very slowly")
    os.environ["DWPOSE_ONNXRT_CHECKED"] = '1'

class DWPose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            detect_hand=(["enable", "disable"], {"default": "enable"}),
            detect_body=(["enable", "disable"], {"default": "enable"}),
            detect_face=(["enable", "disable"], {"default": "enable"}),
            bbox_detector=(["yolox_s.onnx", "yolox_m.onnx", "yolox_l.onnx"], {"default": "yolox_l.onnx"})
        )
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_pose"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses"

    def estimate_pose(self, image, detect_hand, detect_body, detect_face, resolution=512, bbox_detector="yolox_l.onnx", **kwargs):

        detect_hand = detect_hand == "enable"
        detect_body = detect_body == "enable"
        detect_face = detect_face == "enable"

        self.openpose_json = None
        model = DwposeDetector.from_pretrained(DWPOSE_MODEL_NAME, cache_dir=annotator_ckpts_path, det_filename=bbox_detector)
        
        def cb(image, **kwargs):
            result = model(image, **kwargs)
            self.openpose_json = result[1]
            return result[0]
        
        print(f"\nDWPose: Using {bbox_detector} for bbox detection and dw-ll_ucoco_384.onnx for pose estimation")
        out = common_annotator_call(cb, image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body, image_and_json=True, resolution=resolution)
        del model
        return {
            'ui': { "openpose_json": [self.openpose_json] },
            "result": (out, )
        }

NODE_CLASS_MAPPINGS = {
    "DWPreprocessor": DWPose_Preprocessor
}
NODE_DISPLAY_CLASS_MAPPINGS = {
    "DWPreprocessor": "DWPose Pose Recognition"
}