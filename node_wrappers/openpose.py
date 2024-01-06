from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management
import json

class OpenPose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            detect_hand = (["enable", "disable"], {"default": "enable"}),
            detect_body = (["enable", "disable"], {"default": "enable"}),
            detect_face = (["enable", "disable"], {"default": "enable"})
        )
        
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    FUNCTION = "estimate_pose"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses Estimators"

    def estimate_pose(self, image, detect_hand, detect_body, detect_face, resolution=512, **kwargs):
        from controlnet_aux.open_pose import OpenposeDetector

        detect_hand = detect_hand == "enable"
        detect_body = detect_body == "enable"
        detect_face = detect_face == "enable"

        model = OpenposeDetector.from_pretrained().to(model_management.get_torch_device())        
        self.openpose_dicts = []
        def func(image, **kwargs):
            pose_img, openpose_dict = model(image, **kwargs)
            self.openpose_dicts.append(openpose_dict)
            return pose_img
        
        out = common_annotator_call(func, image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body, image_and_json=True, resolution=resolution)
        del model
        return {
            'ui': { "openpose_json": [json.dumps(self.openpose_dicts, indent=4)] },
            "result": (out, self.openpose_dicts)
        }

NODE_CLASS_MAPPINGS = {
    "OpenposePreprocessor": OpenPose_Preprocessor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenposePreprocessor": "OpenPose Pose",
}