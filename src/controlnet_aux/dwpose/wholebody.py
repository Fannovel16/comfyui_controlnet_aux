# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np

from .dw_onnx.cv_ox_det import inference_detector as inference_onnx_yolox
from .dw_onnx.cv_ox_yolo_nas import inference_detector as inference_onnx_yolo_nas
from .dw_onnx.cv_ox_pose import inference_pose as inference_onnx_pose

from .dw_torchscript.jit_det import inference_detector as inference_jit_yolox
from .dw_torchscript.jit_pose import inference_pose as inference_jit_pose

from typing import List, Optional
from .types import PoseResult, BodyResult, Keypoint
from timeit import default_timer
import os
from controlnet_aux.dwpose.util import guess_onnx_input_shape_dtype, get_model_type, get_ort_providers, is_model_torchscript
import torch
import torch.utils.benchmark.utils.timer as torch_timer

class Wholebody:
    def __init__(self, det_model_path: Optional[str] = None, pose_model_path: Optional[str] = None, torchscript_device="cuda"):
        self.det_filename = det_model_path and os.path.basename(det_model_path)
        self.pose_filename = pose_model_path and os.path.basename(pose_model_path)
        self.det, self.pose = None, None
        # return type: None ort cv2 torchscript
        self.det_model_type = get_model_type("DWPose",self.det_filename)
        self.pose_model_type = get_model_type("DWPose",self.pose_filename)
        # Always loads to CPU to avoid building OpenCV.
        cv2_device = 'cpu'
        cv2_backend = cv2.dnn.DNN_BACKEND_OPENCV if cv2_device == 'cpu' else cv2.dnn.DNN_BACKEND_CUDA
        # You need to manually build OpenCV through cmake to work with your GPU.
        cv2_providers = cv2.dnn.DNN_TARGET_CPU if cv2_device == 'cpu' else cv2.dnn.DNN_TARGET_CUDA
        ort_providers = get_ort_providers()

        if self.det_model_type is None:
            pass
        elif self.det_model_type == "ort":
            try:
                import onnxruntime as ort
                self.det = ort.InferenceSession(det_model_path, providers=ort_providers)
            except:
                print(f"Failed to load onnxruntime with {self.det.get_providers()}.\nPlease change EP_list in the config.yaml and restart ComfyUI")
                self.det = ort.InferenceSession(det_model_path, providers=["CPUExecutionProvider"])
        elif self.det_model_type == "cv2":
            try:
                self.det = cv2.dnn.readNetFromONNX(det_model_path)
                self.det.setPreferableBackend(cv2_backend)
                self.det.setPreferableTarget(cv2_providers)
            except:
                print("TopK operators may not work on your OpenCV, try use onnxruntime with CPUExecutionProvider")
                try:
                    import onnxruntime as ort
                    self.det = ort.InferenceSession(det_model_path, providers=["CPUExecutionProvider"])
                except:
                    print(f"Failed to load {det_model_path}, you can use other models instead")
        else:
            self.det = torch.jit.load(det_model_path)
            self.det.to(torchscript_device)

        if self.pose_model_type is None:
            pass
        elif self.pose_model_type == "ort":
            try:
                import onnxruntime as ort
                self.pose = ort.InferenceSession(pose_model_path, providers=ort_providers)
            except:
                print(f"Failed to load onnxruntime with {self.pose.get_providers()}.\nPlease change EP_list in the config.yaml and restart ComfyUI")
                self.pose = ort.InferenceSession(pose_model_path, providers=["CPUExecutionProvider"])
        elif self.pose_model_type == "cv2":
            self.pose = cv2.dnn.readNetFromONNX(pose_model_path)
            self.pose.setPreferableBackend(cv2_backend)
            self.pose.setPreferableTarget(cv2_providers)
        else:
            self.pose = torch.jit.load(pose_model_path)
            self.pose.to(torchscript_device)
        
        if self.pose_filename is not None:
            self.pose_input_size, _ = guess_onnx_input_shape_dtype(self.pose_filename)

    def __call__(self, oriImg) -> Optional[np.ndarray]:
        
        if is_model_torchscript(self.det):
            det_start = torch_timer.timer()
            det_result = inference_jit_yolox(self.det, oriImg, detect_classes=[0])
            print(f"DWPose: Bbox {((torch_timer.timer() - det_start) * 1000):.2f}ms")
        else:
            det_start = default_timer()
            if "yolox" in self.det_filename:
                det_result = inference_onnx_yolox(self.det, oriImg, detect_classes=[0], dtype=np.float32)
            else:
                #FP16 and INT8 YOLO NAS accept uint8 input
                det_result = inference_onnx_yolo_nas(self.det, oriImg, detect_classes=[0], dtype=np.uint8)
            print(f"DWPose: Bbox {((default_timer() - det_start) * 1000):.2f}ms")
        if (det_result is None) or (det_result.shape[0] == 0):
            return None

        if is_model_torchscript(self.pose):
            pose_start = torch_timer.timer()
            keypoints, scores = inference_jit_pose(self.pose, det_result, oriImg, self.pose_input_size)
            print(f"DWPose: Pose {((torch_timer.timer() - pose_start) * 1000):.2f}ms on {det_result.shape[0]} people\n")
        else:
            pose_start = default_timer()
            _, pose_onnx_dtype = guess_onnx_input_shape_dtype(self.pose_filename)
            keypoints, scores = inference_onnx_pose(self.pose, det_result, oriImg, self.pose_input_size, dtype=pose_onnx_dtype)
            print(f"DWPose: Pose {((default_timer() - pose_start) * 1000):.2f}ms on {det_result.shape[0]} people\n")

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        return keypoints_info

    @staticmethod
    def format_result(keypoints_info: Optional[np.ndarray]) -> List[PoseResult]:
        def format_keypoint_part(
            part: np.ndarray,
        ) -> Optional[List[Optional[Keypoint]]]:
            keypoints = [
                Keypoint(x, y, score, i) if score >= 0.3 else None
                for i, (x, y, score) in enumerate(part)
            ]
            return (
                None if all(keypoint is None for keypoint in keypoints) else keypoints
            )

        def total_score(keypoints: Optional[List[Optional[Keypoint]]]) -> float:
            return (
                sum(keypoint.score for keypoint in keypoints if keypoint is not None)
                if keypoints is not None
                else 0.0
            )

        pose_results = []
        if keypoints_info is None:
            return pose_results

        for instance in keypoints_info:
            body_keypoints = format_keypoint_part(instance[:18]) or ([None] * 18)
            left_hand = format_keypoint_part(instance[92:113])
            right_hand = format_keypoint_part(instance[113:134])
            face = format_keypoint_part(instance[24:92])

            # Openpose face consists of 70 points in total, while DWPose only
            # provides 68 points. Padding the last 2 points.
            if face is not None:
                # left eye
                face.append(body_keypoints[14])
                # right eye
                face.append(body_keypoints[15])

            body = BodyResult(
                body_keypoints, total_score(body_keypoints), len(body_keypoints)
            )
            pose_results.append(PoseResult(body, left_hand, right_hand, face))

        return pose_results