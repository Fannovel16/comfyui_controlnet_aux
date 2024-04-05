import folder_paths
import json
import os
import numpy as np
import cv2
from PIL import ImageColor
from einops import rearrange
import torch
from scipy.special import comb
import itertools

"""
Format of POSE_KEYPOINT (AP10K keypoints):
[{
        "version": "ap10k",
        "animals": [
            [[x1, y1, 1], [x2, y2, 1],..., [x17, y17, 1]],
            [[x1, y1, 1], [x2, y2, 1],..., [x17, y17, 1]],
            ...
        ],
        "canvas_height": 512,
        "canvas_width": 768
},...]
Format of POSE_KEYPOINT (OpenPose keypoints):
[{
    "people": [
        {
            'pose_keypoints_2d': [[x1, y1, 1], [x2, y2, 1],..., [x17, y17, 1]]
            "face_keypoints_2d": [[x1, y1, 1], [x2, y2, 1],..., [x68, y68, 1]],
            "hand_left_keypoints_2d": [[x1, y1, 1], [x2, y2, 1],..., [x21, y21, 1]],
            "hand_right_keypoints_2d":[[x1, y1, 1], [x2, y2, 1],..., [x21, y21, 1]],
        }
    ],
    "canvas_height": canvas_height,
    "canvas_width": canvas_width,
},...]
"""

class SavePoseKpsAsJsonFile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_kps": ("POSE_KEYPOINT",),
                "filename_prefix": ("STRING", {"default": "PoseKeypoint"})
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "save_pose_kps"
    OUTPUT_NODE = True
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
    def save_pose_kps(self, pose_kps, filename_prefix):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir, pose_kps[0]["canvas_width"], pose_kps[0]["canvas_height"])
        file = f"{filename}_{counter:05}.json"
        with open(os.path.join(full_output_folder, file), 'w') as f:
            json.dump(pose_kps , f)
        return {}

#COCO-Wholebody doesn't have eyebrows as it inherits 68 keypoints format
#Perhaps eyebrows can be estimated tho
FACIAL_PARTS = ["skin", "left_eye", "right_eye", "nose", "upper_lip", "inner_mouth", "lower_lip"]
LAPA_COLORS = dict(
    skin="rgb(0, 153, 255)",
    left_eye="rgb(0, 204, 153)",
    right_eye="rgb(255, 153, 0)",
    nose="rgb(255, 102, 255)",
    upper_lip="rgb(102, 0, 51)",
    inner_mouth="rgb(255, 204, 255)",
    lower_lip="rgb(255, 0, 102)"
)

#One-based index
def kps_idxs(start, end):
    step = -1 if start > end else 1
    return list(range(start-1, end+1-1, step))

#Source: https://www.researchgate.net/profile/Fabrizio-Falchi/publication/338048224/figure/fig1/AS:837860722741255@1576772971540/68-facial-landmarks.jpg
FACIAL_PART_RANGES = dict(
    skin=kps_idxs(1, 17) + kps_idxs(27, 18),
    nose=kps_idxs(28, 36),
    left_eye=kps_idxs(37, 42),
    right_eye=kps_idxs(43, 48),
    upper_lip=kps_idxs(49, 55) + kps_idxs(65, 61),
    lower_lip=kps_idxs(61, 68),
    inner_mouth=kps_idxs(61, 65) + kps_idxs(55, 49)
)

def is_normalized(keypoints) -> bool:
    point_normalized = [
        0 <= np.abs(k[0]) <= 1 and 0 <= np.abs(k[1]) <= 1 
        for k in keypoints 
        if k is not None
    ]
    if not point_normalized:
        return False
    return np.all(point_normalized)

class FacialPartColoringFromPoseKps:
    @classmethod
    def INPUT_TYPES(s):
        input_types = {
            "required": {"pose_kps": ("POSE_KEYPOINT",), "mode": (["point", "polygon"], {"default": "polygon"})}
        }
        for facial_part in FACIAL_PARTS: 
            input_types["required"][facial_part] = ("STRING", {"default": LAPA_COLORS[facial_part], "multiline": False})
        return input_types
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "colorize"
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"
    def colorize(self, pose_kps, mode, **facial_part_colors):
        pose_frames = pose_kps
        np_frames = [self.draw_kps(pose_frame, mode, **facial_part_colors) for pose_frame in pose_frames]
        np_frames = np.stack(np_frames, axis=0)
        return (torch.from_numpy(np_frames).float() / 255.,)
            
    def draw_kps(self, pose_frame, mode, **facial_part_colors):
        width, height = pose_frame["canvas_width"], pose_frame["canvas_height"]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for person, part_name in itertools.product(pose_frame["people"], FACIAL_PARTS):
            n = len(person["face_keypoints_2d"]) // 3
            facial_kps = rearrange(np.array(person["face_keypoints_2d"]), "(n c) -> n c", n=n, c=3)[:, :2]
            if is_normalized(facial_kps):
                facial_kps *= (width, height)
            facial_kps = facial_kps.astype(np.int32)
            part_color = ImageColor.getrgb(facial_part_colors[part_name])[:3]
            part_contours = facial_kps[FACIAL_PART_RANGES[part_name], :]
            if mode == "point":
                for pt in part_contours:
                    cv2.circle(canvas, pt, radius=2, color=part_color, thickness=-1)
            else:
                cv2.fillPoly(canvas, pts=[part_contours], color=part_color)
        return canvas

NODE_CLASS_MAPPINGS = {
    "SavePoseKpsAsJsonFile": SavePoseKpsAsJsonFile,
    "FacialPartColoringFromPoseKps": FacialPartColoringFromPoseKps
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SavePoseKpsAsJsonFile": "Save Pose Keypoints",
    "FacialPartColoringFromPoseKps": "Colorize Facial Parts from PoseKPS"
}
