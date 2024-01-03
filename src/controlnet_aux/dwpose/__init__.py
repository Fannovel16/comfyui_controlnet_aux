# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th Edited by ControlNet (Improved JSON serialization/deserialization, and lots of bug fixs)
# This preprocessor is licensed by CMU for non-commercial use only.
import torch.utils.benchmark as benchmark
benchmark.timer()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import numpy as np
from . import util
from .body import Body, BodyResult, Keypoint
from .hand import Hand
from .face import Face
from .types import PoseResult, HandResult, FaceResult
from huggingface_hub import hf_hub_download
from .wholebody import Wholebody
import warnings
from controlnet_aux.util import HWC3, resize_image_with_pad, common_input_validate, custom_hf_download
import cv2
from PIL import Image
from .animalpose import AnimalPoseImage

from typing import Tuple, List, Callable, Union, Optional

def draw_poses(poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        if draw_body:
            canvas = util.draw_bodypose(canvas, pose.body.keypoints)

        if draw_hand:
            canvas = util.draw_handpose(canvas, pose.left_hand)
            canvas = util.draw_handpose(canvas, pose.right_hand)

        if draw_face:
            canvas = util.draw_facepose(canvas, pose.face)

    return canvas


def decode_json_as_poses(json_string: str, normalize_coords: bool = False) -> Tuple[List[PoseResult], int, int]:
    """ Decode the json_string complying with the openpose JSON output format
    to poses that controlnet recognizes.
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md

    Args:
        json_string: The json string to decode.
        normalize_coords: Whether to normalize coordinates of each keypoint by canvas height/width.
                          `draw_pose` only accepts normalized keypoints. Set this param to True if
                          the input coords are not normalized.
    
    Returns:
        poses
        canvas_height
        canvas_width                      
    """
    pose_json = json.loads(json_string)
    height = pose_json['canvas_height']
    width = pose_json['canvas_width']

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    def decompress_keypoints(numbers: Optional[List[float]]) -> Optional[List[Optional[Keypoint]]]:
        if not numbers:
            return None
        
        assert len(numbers) % 3 == 0

        def create_keypoint(x, y, c):
            if c < 1.0:
                return None
            keypoint = Keypoint(x, y)
            return keypoint

        return [
            create_keypoint(x, y, c)
            for x, y, c in chunks(numbers, n=3)
        ]
    
    return (
        [
            PoseResult(
                body=BodyResult(keypoints=decompress_keypoints(pose.get('pose_keypoints_2d'))),
                left_hand=decompress_keypoints(pose.get('hand_left_keypoints_2d')),
                right_hand=decompress_keypoints(pose.get('hand_right_keypoints_2d')),
                face=decompress_keypoints(pose.get('face_keypoints_2d'))
            )
            for pose in pose_json['people']
        ],
        height,
        width,
    )


def encode_poses_as_dict(poses: List[PoseResult], canvas_height: int, canvas_width: int) -> str:
    """ Encode the pose as a dict following openpose JSON output format:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
    """
    def compress_keypoints(keypoints: Union[List[Keypoint], None]) -> Union[List[float], None]:
        if not keypoints:
            return None
        
        return [
            value
            for keypoint in keypoints
            for value in (
                [float(keypoint.x), float(keypoint.y), 1.0]
                if keypoint is not None
                else [0.0, 0.0, 0.0]
            )
        ]

    return {
        'people': [
            {
                'pose_keypoints_2d': compress_keypoints(pose.body.keypoints),
                "face_keypoints_2d": compress_keypoints(pose.face),
                "hand_left_keypoints_2d": compress_keypoints(pose.left_hand),
                "hand_right_keypoints_2d":compress_keypoints(pose.right_hand),
            }
            for pose in poses
        ],
        'canvas_height': canvas_height,
        'canvas_width': canvas_width,
    }

global_cached_dwpose = Wholebody()

class DwposeDetector:
    """
    A class for detecting human poses in images using the Dwpose model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """
    def __init__(self, dw_pose_estimation):
        self.dw_pose_estimation = dw_pose_estimation
    
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, pretrained_det_model_or_path=None, det_filename=None, pose_filename=None, torchscript_device="cuda"):
        global global_cached_dwpose
        pretrained_det_model_or_path = pretrained_det_model_or_path or pretrained_model_or_path
        det_filename = det_filename or "yolox_l.onnx"
        pose_filename = pose_filename or "dw-ll_ucoco_384.onnx"
        det_model_path = custom_hf_download(pretrained_det_model_or_path, det_filename)
        pose_model_path = custom_hf_download(pretrained_model_or_path, pose_filename)
        
        print(f"\nDWPose: Using {det_filename} for bbox detection and {pose_filename} for pose estimation")
        if global_cached_dwpose.det is None or global_cached_dwpose.det_filename != det_filename:
            t = Wholebody(det_model_path, None, torchscript_device=torchscript_device)
            t.pose = global_cached_dwpose.pose
            t.pose_filename = global_cached_dwpose.pose
            global_cached_dwpose = t
        
        if global_cached_dwpose.pose is None or global_cached_dwpose.pose_filename != pose_filename:
            t = Wholebody(None, pose_model_path, torchscript_device=torchscript_device)
            t.det = global_cached_dwpose.det
            t.det_filename = global_cached_dwpose.det_filename
            global_cached_dwpose = t
        return cls(global_cached_dwpose)

    def detect_poses(self, oriImg) -> List[PoseResult]:
        with torch.no_grad():
            keypoints_info = self.dw_pose_estimation(oriImg.copy())
            return Wholebody.format_result(keypoints_info)
    
    def __call__(self, input_image, detect_resolution=512, include_body=True, include_hand=False, include_face=False, hand_and_face=None, output_type="pil", image_and_json=False, upscale_method="INTER_CUBIC", **kwargs):
        if hand_and_face is not None:
            warnings.warn("hand_and_face is deprecated. Use include_hand and include_face instead.", DeprecationWarning)
            include_hand = hand_and_face
            include_face = hand_and_face

        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        poses = self.detect_poses(input_image)
        canvas = draw_poses(poses, input_image.shape[0], input_image.shape[1], draw_body=include_body, draw_hand=include_hand, draw_face=include_face) 
        detected_map = HWC3(remove_pad(canvas))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        
        if image_and_json:
            return (detected_map, encode_poses_as_dict(poses, detected_map.shape[0], detected_map.shape[1]))
        
        return detected_map

global_cached_animalpose = AnimalPoseImage()
class AnimalposeDetector:
    """
    A class for detecting animal poses in images using the RTMPose AP10k model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """
    def __init__(self, animal_pose_estimation):
        self.animal_pose_estimation = animal_pose_estimation
    
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, pretrained_det_model_or_path=None, det_filename="yolox_l.onnx", pose_filename="dw-ll_ucoco_384.onnx", torchscript_device="cuda"):
        global global_cached_animalpose
        det_model_path = custom_hf_download(pretrained_det_model_or_path, det_filename)
        pose_model_path = custom_hf_download(pretrained_model_or_path, pose_filename)
        
        print(f"\nAnimalPose: Using {det_filename} for bbox detection and {pose_filename} for pose estimation")
        if global_cached_animalpose.det is None or global_cached_animalpose.det_filename != det_filename:
            t = AnimalPoseImage(det_model_path, None, torchscript_device=torchscript_device)
            t.pose = global_cached_animalpose.pose
            t.pose_filename = global_cached_animalpose.pose
            global_cached_animalpose = t
        
        if global_cached_animalpose.pose is None or global_cached_animalpose.pose_filename != pose_filename:
            t = AnimalPoseImage(None, pose_model_path, torchscript_device=torchscript_device)
            t.det = global_cached_animalpose.det
            t.det_filename = global_cached_animalpose.det_filename
            global_cached_animalpose = t
        return cls(global_cached_animalpose)
    
    def __call__(self, input_image, detect_resolution=512, output_type="pil", image_and_json=False, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        detected_map, openpose_dict = self.animal_pose_estimation(input_image)
        detected_map = remove_pad(detected_map)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        
        if image_and_json:
            return (detected_map, openpose_dict)

        return detected_map