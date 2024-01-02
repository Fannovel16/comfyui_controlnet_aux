from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management
import os, sys
import subprocess, threading

#Ref: https://github.com/ltdrdata/ComfyUI-Manager/blob/284e90dc8296a2e1e4f14b4b2d10fba2f52f0e53/__init__.py#L14
def handle_stream(stream, prefix):
    for line in stream:
        print(prefix, line, end="")


def run_script(cmd, cwd='.'):
    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    stdout_thread = threading.Thread(target=handle_stream, args=(process.stdout, ""))
    stderr_thread = threading.Thread(target=handle_stream, args=(process.stderr, "[!]"))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()

def install_deps():
    try:
        import mediapipe
    except ImportError:
        run_script([sys.executable, '-s', '-m', 'pip', 'install', 'mediapipe'])
        run_script([sys.executable, '-s', '-m', 'pip', 'install', '--upgrade', 'protobuf'])

class Media_Pipe_Face_Mesh_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            max_faces=("INT", {"default": 10, "min": 1, "max": 50, "step": 1}), #Which image has more than 50 detectable faces?
            min_confidence=("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01})
        )
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses Estimators"

    def detect(self, image, max_faces, min_confidence, resolution=512):
        #Ref: https://github.com/Fannovel16/comfy_controlnet_preprocessors/issues/70#issuecomment-1677967369
        install_deps()
        from controlnet_aux.mediapipe_face import MediapipeFaceDetector
        return (common_annotator_call(MediapipeFaceDetector(), image, max_faces=max_faces, min_confidence=min_confidence, resolution=resolution), )

NODE_CLASS_MAPPINGS = {
    "MediaPipe-FaceMeshPreprocessor": Media_Pipe_Face_Mesh_Preprocessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipe-FaceMeshPreprocessor": "MediaPipe Face Mesh"
}