from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME, DWPOSE_MODEL_NAME, create_node_input_types
import comfy.model_management as model_management
import numpy as np
import warnings
from controlnet_aux.dwpose import DwposeDetector, AnimalposeDetector
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
        input_types = create_node_input_types(
            detect_hand=(["enable", "disable"], {"default": "enable"}),
            detect_body=(["enable", "disable"], {"default": "enable"}),
            detect_face=(["enable", "disable"], {"default": "enable"})
        )
        input_types["optional"] = {
            **input_types["optional"],
            "bbox_detector": (
                ["yolo_nas_l_fp16.onnx", "yolo_nas_m_fp16.onnx", "yolo_nas_s_fp16.onnx", "yolox_l.onnx", "yolox_m.onnx", "yolox_s.onnx"], 
                {"default": "yolox_l.onnx"}
            ),
            "pose_estimator": (["dw-ll_ucoco_384.onnx", "dw-ll_ucoco.onnx", "dw-mm_ucoco.onnx", "dw-ss_ucoco.onnx"], {"default": "dw-ll_ucoco_384.onnx"})
        }
        return input_types
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_pose"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses"

    def estimate_pose(self, image, detect_hand, detect_body, detect_face, resolution=512, bbox_detector="yolox_l.onnx", pose_estimator="dw-ll_ucoco_384.onnx", **kwargs):
        yolo_repo = DWPOSE_MODEL_NAME
        if (bbox_detector != "yolox_l.onnx") and ("yolox" in bbox_detector):
            yolo_repo = "hr16/yolox-onnx"
        elif "yolo_nas" in bbox_detector:
            yolo_repo = "hr16/yolo-nas-fp16"
        detect_hand = detect_hand == "enable"
        detect_body = detect_body == "enable"
        detect_face = detect_face == "enable"
        self.openpose_json = None
        
        model = DwposeDetector.from_pretrained(
            DWPOSE_MODEL_NAME if pose_estimator == "dw-ll_ucoco_384.onnx" else "hr16/UnJIT-DWPose",
            yolo_repo,
            cache_dir=annotator_ckpts_path, det_filename=bbox_detector, pose_filename=pose_estimator
        )
        
        def func(image, **kwargs):
            result = model(image, **kwargs)
            self.openpose_json = result[1]
            return result[0]
        
        print(f"\nDWPose: Using {bbox_detector} for bbox detection and {pose_estimator} for pose estimation")
        out = common_annotator_call(func, image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body, image_and_json=True, resolution=resolution)
        del model
        return {
            'ui': { "openpose_json": [self.openpose_json] },
            "result": (out, )
        }

class AnimalPose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            bbox_detector = (
                ["yolo_nas_l_fp16.onnx", "yolo_nas_m_fp16.onnx", "yolo_nas_s_fp16.onnx", "yolox_l.onnx", "yolo_nas_m.onnx", "yolox_s.onnx"], 
                {"default": "yolox_l.onnx"}
            ),
            pose_estimator = (["rtmpose-m_ap10k_256.onnx"], {"default": "rtmpose-m_ap10k_256.onnx"})
        )
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_pose"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses"

    def estimate_pose(self, image, resolution=512, bbox_detector="yolox_l.onnx", pose_estimator="rtmpose-m_ap10k_256.onnx", **kwargs):
        yolo_repo = DWPOSE_MODEL_NAME
        if (bbox_detector != "yolox_l.onnx") and ("yolox" in bbox_detector):
            yolo_repo = "hr16/yolox-onnx"
        elif "yolo_nas" in bbox_detector:
            yolo_repo = "hr16/yolo-nas-fp16"
        model = AnimalposeDetector.from_pretrained(
            DWPOSE_MODEL_NAME if pose_estimator == "dw-ll_ucoco_384.onnx" else "hr16/UnJIT-DWPose",
            yolo_repo,
            cache_dir=annotator_ckpts_path, det_filename=bbox_detector, pose_filename=pose_estimator
        )
        print(f"\nAnimalPose: Using {bbox_detector} for bbox detection and {pose_estimator} for pose estimation")

        def func(image, **kwargs):
            result = model(image, **kwargs)
            self.openpose_json = result[1]
            return result[0]

        out = common_annotator_call(func, image, image_and_json=True, resolution=resolution)
        del model
        return {
            'ui': { "openpose_json": [self.openpose_json] },
            "result": (out, )
        }

NODE_CLASS_MAPPINGS = {
    "DWPreprocessor": DWPose_Preprocessor,
    "AnimalPosePreprocessor": AnimalPose_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DWPreprocessor": "DWPose Estimation",
    "AnimalPosePreprocessor": "Animal Pose Estimation (AP10K)"
}