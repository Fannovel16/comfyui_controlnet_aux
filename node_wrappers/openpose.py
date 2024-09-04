from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management
import json

class OpenPose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            detect_hand=INPUT.COMBO(["enable", "disable"]),
            detect_body=INPUT.COMBO(["enable", "disable"]),
            detect_face=INPUT.COMBO(["enable", "disable"]),
            resolution=INPUT.RESOLUTION(),
            scale_stick_for_xinsr_cn=INPUT.COMBO(["disable", "enable"])
        )
        
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    FUNCTION = "estimate_pose"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses Estimators"

    def estimate_pose(self, image, detect_hand="enable", detect_body="enable", detect_face="enable", scale_stick_for_xinsr_cn="disable", resolution=512, **kwargs):
        from custom_controlnet_aux.open_pose import OpenposeDetector

        detect_hand = detect_hand == "enable"
        detect_body = detect_body == "enable"
        detect_face = detect_face == "enable"
        scale_stick_for_xinsr_cn = scale_stick_for_xinsr_cn == "enable"

        model = OpenposeDetector.from_pretrained().to(model_management.get_torch_device())        
        self.openpose_dicts = []
        def func(image, **kwargs):
            pose_img, openpose_dict = model(image, **kwargs)
            self.openpose_dicts.append(openpose_dict)
            return pose_img
        
        out = common_annotator_call(func, image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body, image_and_json=True, xinsr_stick_scaling=scale_stick_for_xinsr_cn, resolution=resolution)
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