import folder_paths
import json
import os

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

NODE_CLASS_MAPPINGS = {
    "SavePoseKpsAsJsonFile": SavePoseKpsAsJsonFile
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SavePoseKpsAsJsonFile": "Save Pose Keypoints"
}