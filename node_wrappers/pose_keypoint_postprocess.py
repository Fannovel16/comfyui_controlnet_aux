import folder_paths
import json
import os
import numpy as np
import cv2
from PIL import ImageColor
from einops import rearrange
import torch
import itertools

from ..src.custom_controlnet_aux.dwpose import draw_poses, draw_animalposes, decode_json_as_poses


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

# https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/.github/media/keypoints_pose_18.png
BODY_PART_INDEXES = {
    "Head": (16, 14, 0, 15, 17),
    "Neck": (0, 1),
    "Shoulder": (2, 5),
    "Torso": (2, 5, 8, 11),
    "RArm": (2, 3),
    "RForearm": (3, 4),
    "LArm": (5, 6),
    "LForearm": (6, 7),
    "RThigh": (8, 9),
    "RLeg": (9, 10),
    "LThigh": (11, 12),
    "LLeg": (12, 13)
}
BODY_PART_DEFAULT_W_H = {
    "Head": "256, 256",
    "Neck": "100, 100",
    "Shoulder": '',
    "Torso": "350, 450",
    "RArm": "128, 256",
    "RForearm": "128, 256",
    "LArm": "128, 256",
    "LForearm": "128, 256",
    "RThigh": "128, 256",
    "RLeg": "128, 256",
    "LThigh": "128, 256",
    "LLeg": "128, 256"
}

class SinglePersonProcess:
    @classmethod 
    def sort_and_get_max_people(s, pose_kps):
        for idx in range(len(pose_kps)):
            pose_kps[idx]["people"] = sorted(pose_kps[idx]["people"], key=lambda person:person["pose_keypoints_2d"][0])
        return pose_kps, max(len(frame["people"]) for frame in pose_kps)
    
    def __init__(self, pose_kps, person_idx=0) -> None:
        self.width, self.height = pose_kps[0]["canvas_width"], pose_kps[0]["canvas_height"]
        self.poses = [
            self.normalize(pose_frame["people"][person_idx]["pose_keypoints_2d"])
            if person_idx < len(pose_frame["people"]) 
            else None
            for pose_frame in pose_kps
        ]
    
    def normalize(self, pose_kps_2d):
        n = len(pose_kps_2d) // 3
        pose_kps_2d = rearrange(np.array(pose_kps_2d), "(n c) -> n c", n=n, c=3)
        pose_kps_2d[np.argwhere(pose_kps_2d[:,2]==0), :] = np.iinfo(np.int32).max // 2 #Safe large value
        pose_kps_2d = pose_kps_2d[:, :2]
        if is_normalized(pose_kps_2d):
            pose_kps_2d *= (self.width, self.height)
        return pose_kps_2d
    
    def get_xyxy_bboxes(self, part_name, bbox_size=(128, 256)):
        width, height = bbox_size
        xyxy_bboxes = {}
        for idx, pose in enumerate(self.poses):
            if pose is None:
                xyxy_bboxes[idx] = (np.iinfo(np.int32).max // 2,) * 4
                continue
            pts = pose[BODY_PART_INDEXES[part_name], :]

            #top_left = np.min(pts[:,0]), np.min(pts[:,1])
            #bottom_right = np.max(pts[:,0]), np.max(pts[:,1])
            #pad_width = np.maximum(width - (bottom_right[0]-top_left[0]), 0) / 2
            #pad_height = np.maximum(height - (bottom_right[1]-top_left[1]), 0) / 2
            #xyxy_bboxes.append((
            #    top_left[0] - pad_width, top_left[1] - pad_height,
            #    bottom_right[0] + pad_width, bottom_right[1] + pad_height,
            #))

            x_mid, y_mid = np.mean(pts[:, 0]), np.mean(pts[:, 1])
            xyxy_bboxes[idx] = (
                x_mid - width/2, y_mid - height/2,
                x_mid + width/2, y_mid + height/2 
            )
        return xyxy_bboxes

class UpperBodyTrackingFromPoseKps:
    PART_NAMES = ["Head", "Neck", "Shoulder", "Torso", "RArm", "RForearm", "LArm", "LForearm"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_kps": ("POSE_KEYPOINT",),
                "id_include": ("STRING", {"default": '', "multiline": False}),
                **{part_name + "_width_height": ("STRING", {"default": BODY_PART_DEFAULT_W_H[part_name], "multiline": False}) for part_name in s.PART_NAMES}
            }
        }

    RETURN_TYPES = ("TRACKING", "STRING")
    RETURN_NAMES = ("tracking", "prompt")
    FUNCTION = "convert"
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"

    def convert(self, pose_kps, id_include, **parts_width_height):
        parts_width_height = {part_name.replace("_width_height", ''): value for part_name, value in parts_width_height.items()}
        enabled_part_names = [part_name for part_name in self.PART_NAMES if len(parts_width_height[part_name].strip())]
        tracked = {part_name: {} for part_name in enabled_part_names}
        id_include = id_include.strip()
        id_include = list(map(int, id_include.split(','))) if len(id_include) else []
        prompt_string = ''
        pose_kps, max_people = SinglePersonProcess.sort_and_get_max_people(pose_kps)

        for person_idx in range(max_people):
            if len(id_include) and person_idx not in id_include:
                continue
            processor = SinglePersonProcess(pose_kps, person_idx)
            for part_name in enabled_part_names:
                bbox_size = tuple(map(int, parts_width_height[part_name].split(',')))
                part_bboxes = processor.get_xyxy_bboxes(part_name, bbox_size)
                id_coordinates = {idx: part_bbox+(processor.width, processor.height) for idx, part_bbox in part_bboxes.items()}
                tracked[part_name][person_idx] = id_coordinates

        for class_name, class_data in tracked.items():
            for class_id in class_data.keys():
                class_id_str = str(class_id)
                # Use the incoming prompt for each class name and ID
                _class_name = class_name.replace('L', '').replace('R', '').lower()
                prompt_string += f'"{class_id_str}.{class_name}": "({_class_name})",\n'

        return (tracked, prompt_string)


def numpy2torch(np_image: np.ndarray) -> torch.Tensor:
    """ [H, W, C] => [B=1, H, W, C]"""
    return torch.from_numpy(np_image.astype(np.float32) / 255).unsqueeze(0)


class RenderPeopleKps:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kps": ("POSE_KEYPOINT",),
                "render_body": ("BOOLEAN", {"default": True}),
                "render_hand": ("BOOLEAN", {"default": True}),
                "render_face": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"

    def render(self, kps, render_body, render_hand, render_face) -> tuple[np.ndarray]:
        if isinstance(kps, list):
            kps = kps[0]

        poses, _, height, width = decode_json_as_poses(kps)
        np_image = draw_poses(
            poses,
            height,
            width,
            render_body,
            render_hand,
            render_face,
        )
        return (numpy2torch(np_image),)

class RenderAnimalKps:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kps": ("POSE_KEYPOINT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"

    def render(self, kps) -> tuple[np.ndarray]:
        if isinstance(kps, list):
            kps = kps[0]

        _, poses, height, width = decode_json_as_poses(kps)
        np_image = draw_animalposes(poses, height, width)
        return (numpy2torch(np_image),)


NODE_CLASS_MAPPINGS = {
    "SavePoseKpsAsJsonFile": SavePoseKpsAsJsonFile,
    "FacialPartColoringFromPoseKps": FacialPartColoringFromPoseKps,
    "UpperBodyTrackingFromPoseKps": UpperBodyTrackingFromPoseKps,
    "RenderPeopleKps": RenderPeopleKps,
    "RenderAnimalKps": RenderAnimalKps,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SavePoseKpsAsJsonFile": "Save Pose Keypoints",
    "FacialPartColoringFromPoseKps": "Colorize Facial Parts from PoseKPS",
    "UpperBodyTrackingFromPoseKps": "Upper Body Tracking From PoseKps (InstanceDiffusion)",
    "RenderPeopleKps": "Render Pose JSON (Human)",
    "RenderAnimalKps": "Render Pose JSON (Animal)",
}
