from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT, run_script
import comfy.model_management as model_management
import os, sys
import subprocess, threading

def install_deps():
    try:
        import mediapipe
    except ImportError:
        run_script([sys.executable, '-s', '-m', 'pip', 'install', 'mediapipe'])
        run_script([sys.executable, '-s', '-m', 'pip', 'install', '--upgrade', 'protobuf'])

class Media_Pipe_Face_Mesh_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            max_faces=INPUT.INT(default=10, min=1, max=50), #Which image has more than 50 detectable faces?
            min_confidence=INPUT.FLOAT(default=0.5, min=0.1),
            resolution=INPUT.RESOLUTION()
        )
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses Estimators"

    def detect(self, image, max_faces=10, min_confidence=0.5, resolution=512):
        #Ref: https://github.com/Fannovel16/comfy_controlnet_preprocessors/issues/70#issuecomment-1677967369
        install_deps()
        from custom_controlnet_aux.mediapipe_face import MediapipeFaceDetector
        return (common_annotator_call(MediapipeFaceDetector(), image, max_faces=max_faces, min_confidence=min_confidence, resolution=resolution), )

NODE_CLASS_MAPPINGS = {
    "MediaPipe-FaceMeshPreprocessor": Media_Pipe_Face_Mesh_Preprocessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipe-FaceMeshPreprocessor": "MediaPipe Face Mesh"
}