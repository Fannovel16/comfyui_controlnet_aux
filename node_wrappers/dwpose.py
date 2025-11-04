from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management
import numpy as np
import warnings
from ..src.custom_controlnet_aux.dwpose import DwposeDetector, AnimalposeDetector
import os
import json

DWPOSE_MODEL_NAME = "yzd-v/DWPose"
#Trigger startup caching for onnxruntime
GPU_PROVIDERS = ["CUDAExecutionProvider", "DirectMLExecutionProvider", "OpenVINOExecutionProvider", "ROCMExecutionProvider", "CoreMLExecutionProvider", "MIGraphXExecutionProvider"]
def check_ort_gpu():
    try:
        import onnxruntime as ort
        for provider in GPU_PROVIDERS:
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
        os.environ['AUX_ORT_PROVIDERS'] = ''
    os.environ["DWPOSE_ONNXRT_CHECKED"] = '1'

class DWPose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            detect_hand=INPUT.COMBO(["enable", "disable"]),
            detect_body=INPUT.COMBO(["enable", "disable"]),
            detect_face=INPUT.COMBO(["enable", "disable"]),
            resolution=INPUT.RESOLUTION(),
            bbox_detector=INPUT.COMBO(
                ["None"] + ["yolox_l.torchscript.pt", "yolox_l.onnx", "yolo_nas_l_fp16.onnx", "yolo_nas_m_fp16.onnx", "yolo_nas_s_fp16.onnx"],
                default="yolox_l.onnx"
            ),
            pose_estimator=INPUT.COMBO(
                ["dw-ll_ucoco_384_bs5.torchscript.pt", "dw-ll_ucoco_384.onnx", "dw-ll_ucoco.onnx"],
                default="dw-ll_ucoco_384_bs5.torchscript.pt"
            ),
            scale_stick_for_xinsr_cn=INPUT.COMBO(["disable", "enable"])
        )

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    FUNCTION = "estimate_pose"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses Estimators"

    def estimate_pose(self, image, detect_hand="enable", detect_body="enable", detect_face="enable", resolution=512, bbox_detector="yolox_l.onnx", pose_estimator="dw-ll_ucoco_384.onnx", scale_stick_for_xinsr_cn="disable", **kwargs):
        if bbox_detector == "None":
            yolo_repo = DWPOSE_MODEL_NAME 
        elif bbox_detector == "yolox_l.onnx":
            yolo_repo = DWPOSE_MODEL_NAME
        elif "yolox" in bbox_detector:
            yolo_repo = "hr16/yolox-onnx"
        elif "yolo_nas" in bbox_detector:
            yolo_repo = "hr16/yolo-nas-fp16"
        else:
            raise NotImplementedError(f"Download mechanism for {bbox_detector}")

        if pose_estimator == "dw-ll_ucoco_384.onnx":
            pose_repo = DWPOSE_MODEL_NAME
        elif pose_estimator.endswith(".onnx"):
            pose_repo = "hr16/UnJIT-DWPose"
        elif pose_estimator.endswith(".torchscript.pt"):
            pose_repo = "hr16/DWPose-TorchScript-BatchSize5"
        else:
            raise NotImplementedError(f"Download mechanism for {pose_estimator}")

        model = DwposeDetector.from_pretrained(
            pose_repo,
            yolo_repo,
            det_filename=(None if bbox_detector == "None" else bbox_detector), pose_filename=pose_estimator,
            torchscript_device=model_management.get_torch_device()
        )
        detect_hand = detect_hand == "enable"
        detect_body = detect_body == "enable"
        detect_face = detect_face == "enable"
        scale_stick_for_xinsr_cn = scale_stick_for_xinsr_cn == "enable"
        self.openpose_dicts = []
        def func(image, **kwargs):
            pose_img, openpose_dict = model(image, **kwargs)
            self.openpose_dicts.append(openpose_dict)
            return pose_img

        out = common_annotator_call(func, image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body, image_and_json=True, resolution=resolution, xinsr_stick_scaling=scale_stick_for_xinsr_cn)
        del model
        return {
            'ui': { "openpose_json": [json.dumps(self.openpose_dicts, indent=4)] },
            "result": (out, self.openpose_dicts)
        }

class AnimalPose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            bbox_detector = INPUT.COMBO(
                ["None"] + ["yolox_l.torchscript.pt", "yolox_l.onnx", "yolo_nas_l_fp16.onnx", "yolo_nas_m_fp16.onnx", "yolo_nas_s_fp16.onnx"],
                default="yolox_l.torchscript.pt"
            ),
            pose_estimator = INPUT.COMBO(
                ["rtmpose-m_ap10k_256_bs5.torchscript.pt", "rtmpose-m_ap10k_256.onnx"],
                default="rtmpose-m_ap10k_256_bs5.torchscript.pt"
            ),
            resolution = INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    FUNCTION = "estimate_pose"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses Estimators"

    def estimate_pose(self, image, resolution=512, bbox_detector="yolox_l.onnx", pose_estimator="rtmpose-m_ap10k_256.onnx", **kwargs):
        if bbox_detector == "None":
            yolo_repo = DWPOSE_MODEL_NAME 
        elif bbox_detector == "yolox_l.onnx":
            yolo_repo = DWPOSE_MODEL_NAME
        elif "yolox" in bbox_detector:
            yolo_repo = "hr16/yolox-onnx"
        elif "yolo_nas" in bbox_detector:
            yolo_repo = "hr16/yolo-nas-fp16"
        else:
            raise NotImplementedError(f"Download mechanism for {bbox_detector}")

        if pose_estimator == "dw-ll_ucoco_384.onnx":
            pose_repo = DWPOSE_MODEL_NAME
        elif pose_estimator.endswith(".onnx"):
            pose_repo = "hr16/UnJIT-DWPose"
        elif pose_estimator.endswith(".torchscript.pt"):
            pose_repo = "hr16/DWPose-TorchScript-BatchSize5"
        else:
            raise NotImplementedError(f"Download mechanism for {pose_estimator}")

        model = AnimalposeDetector.from_pretrained(
            pose_repo,
            yolo_repo,
            det_filename=(None if bbox_detector == "None" else bbox_detector), pose_filename=pose_estimator,
            torchscript_device=model_management.get_torch_device()
        )

        self.openpose_dicts = []
        def func(image, **kwargs):
            pose_img, openpose_dict = model(image, **kwargs)
            self.openpose_dicts.append(openpose_dict)
            return pose_img

        out = common_annotator_call(func, image, image_and_json=True, resolution=resolution)
        del model
        return {
            'ui': { "openpose_json": [json.dumps(self.openpose_dicts, indent=4)] },
            "result": (out, self.openpose_dicts)
        }

NODE_CLASS_MAPPINGS = {
    "DWPreprocessor": DWPose_Preprocessor,
    "AnimalPosePreprocessor": AnimalPose_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DWPreprocessor": "DWPose Estimator",
    "AnimalPosePreprocessor": "AnimalPose Estimator (AP10K)"
}
