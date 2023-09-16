from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME, DWPOSE_MODEL_NAME, create_node_input_types
import comfy.model_management as model_management

class DWPose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            detect_hand=(["enable", "disable"], {"default": "enable"}),
            detect_body=(["enable", "disable"], {"default": "enable"}),
            detect_face=(["enable", "disable"], {"default": "enable"})
        )
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_pose"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses"

    def estimate_pose(self, image, detect_hand, detect_body, detect_face, resolution, **kwargs):
        from controlnet_aux.dwpose import DwposeDetector

        detect_hand = detect_hand == "enable"
        detect_body = detect_body == "enable"
        detect_face = detect_face == "enable"

        self.openpose_json = None
        model = DwposeDetector.from_pretrained(DWPOSE_MODEL_NAME, cache_dir=annotator_ckpts_path).to(model_management.get_torch_device())
        
        def cb(image, **kwargs):
            result = model(image, **kwargs)
            self.openpose_json = result[1]
            return result[0]
        
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