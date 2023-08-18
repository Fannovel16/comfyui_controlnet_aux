from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME, DWPOSE_MODEL_NAME
from controlnet_aux.mediapipe_face import MediapipeFaceDetector
import comfy.model_management as model_management

class Media_Pipe_Face_Mesh_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE", ),
                              "max_faces": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}), #Which image has more than 50 detectable faces?
                              "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01})
                            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect"

    CATEGORY = "preprocessors/face_mesh"

    def detect(self, image, max_faces, min_confidence):
        return (common_annotator_call(MediapipeFaceDetector(), image, max_faces=max_faces, min_confidence=min_confidence), )