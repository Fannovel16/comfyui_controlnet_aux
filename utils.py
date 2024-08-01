import torch
import numpy as np
import os
import cv2
import yaml
from pathlib import Path
from enum import Enum
from .log import log
import subprocess
import threading
import comfy
import tempfile
import folder_paths

here = Path(__file__).parent.resolve()

config_path = Path(here, "config.yaml")

if os.path.exists(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    annotator_ckpts_path = str(Path(here, config["annotator_ckpts_path"]))
    TEMP_DIR = config["custom_temp_path"]
    USE_SYMLINKS = config["USE_SYMLINKS"]
    ORT_PROVIDERS = config["EP_list"]

    if USE_SYMLINKS is None or type(USE_SYMLINKS) != bool:
        log.error("USE_SYMLINKS must be a boolean. Using False by default.")
        USE_SYMLINKS = False

    if TEMP_DIR is None:
        TEMP_DIR = tempfile.gettempdir()
    elif not os.path.isdir(TEMP_DIR):
        try:
            os.makedirs(TEMP_DIR)
        except:
            log.error("Failed to create custom temp directory. Using default.")
            TEMP_DIR = tempfile.gettempdir()

    if not os.path.isdir(annotator_ckpts_path):
        try:
            os.makedirs(annotator_ckpts_path)
        except:
            log.error("Failed to create config ckpts directory. Using default.")
            annotator_ckpts_path = str(Path(here, "./ckpts"))
else:
    annotator_ckpts_path = str(Path(here, "./ckpts"))
    TEMP_DIR = tempfile.gettempdir()
    USE_SYMLINKS = False
    ORT_PROVIDERS = ["CUDAExecutionProvider", "DirectMLExecutionProvider", "OpenVINOExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider", "CoreMLExecutionProvider"]

# 使用comfyui的目录结构 Directory structure for using ComfyUI
if "controlnet_ckpts" in folder_paths.folder_names_and_paths:
    annotator_ckpts_path=folder_paths.get_folder_paths('controlnet_ckpts')[0]

os.environ['AUX_ANNOTATOR_CKPTS_PATH'] = os.getenv('AUX_ANNOTATOR_CKPTS_PATH', annotator_ckpts_path)
os.environ['AUX_TEMP_DIR'] = os.getenv('AUX_TEMP_DIR', str(TEMP_DIR))
os.environ['AUX_USE_SYMLINKS'] = os.getenv('AUX_USE_SYMLINKS', str(USE_SYMLINKS))
os.environ['AUX_ORT_PROVIDERS'] = os.getenv('AUX_ORT_PROVIDERS', str(",".join(ORT_PROVIDERS)))

log.info(f"Using ckpts path: {annotator_ckpts_path}")
log.info(f"Using symlinks: {USE_SYMLINKS}")
log.info(f"Using ort providers: {ORT_PROVIDERS}")

# Sync with theoritical limit from Comfy base
# https://github.com/comfyanonymous/ComfyUI/blob/eecd69b53a896343775bcb02a4f8349e7442ffd1/nodes.py#L45
MAX_RESOLUTION=16384

def common_annotator_call(model, tensor_image, input_batch=False, show_pbar=True, **kwargs):
    if "detect_resolution" in kwargs:
        del kwargs["detect_resolution"] #Prevent weird case?

    if "resolution" in kwargs:
        detect_resolution = kwargs["resolution"] if type(kwargs["resolution"]) == int and kwargs["resolution"] >= 64 else 512
        del kwargs["resolution"]
    else:
        detect_resolution = 512

    if input_batch:
        np_images = np.asarray(tensor_image * 255., dtype=np.uint8)
        np_results = model(np_images, output_type="np", detect_resolution=detect_resolution, **kwargs)
        return torch.from_numpy(np_results.astype(np.float32) / 255.0)

    batch_size = tensor_image.shape[0]
    if show_pbar:
        pbar = comfy.utils.ProgressBar(batch_size)
    out_tensor = None
    for i, image in enumerate(tensor_image):
        np_image = np.asarray(image.cpu() * 255., dtype=np.uint8)
        np_result = model(np_image, output_type="np", detect_resolution=detect_resolution, **kwargs)
        out = torch.from_numpy(np_result.astype(np.float32) / 255.0)
        if out_tensor is None:
            out_tensor = torch.zeros(batch_size, *out.shape, dtype=torch.float32)
        out_tensor[i] = out
        if show_pbar:
            pbar.update(1)
    return out_tensor

def create_node_input_types(**extra_kwargs):
    return {
        "required": {
            "image": ("IMAGE",)
        },
        "optional": {
            **extra_kwargs,
            "resolution": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64})
        }
    }

class ResizeMode(Enum):
    """
    Resize modes for ControlNet input images.
    """

    RESIZE = "Just Resize"
    INNER_FIT = "Crop and Resize"
    OUTER_FIT = "Resize and Fill"

    def int_value(self):
        if self == ResizeMode.RESIZE:
            return 0
        elif self == ResizeMode.INNER_FIT:
            return 1
        elif self == ResizeMode.OUTER_FIT:
            return 2
        assert False, "NOTREACHED"

#https://github.com/Mikubill/sd-webui-controlnet/blob/e67e017731aad05796b9615dc6eadce911298ea1/internal_controlnet/external_code.py#L89
#Replaced logger with internal log
def pixel_perfect_resolution(
        image: np.ndarray,
        target_H: int,
        target_W: int,
        resize_mode: ResizeMode,
) -> int:
    """
    Calculate the estimated resolution for resizing an image while preserving aspect ratio.

    The function first calculates scaling factors for height and width of the image based on the target
    height and width. Then, based on the chosen resize mode, it either takes the smaller or the larger
    scaling factor to estimate the new resolution.

    If the resize mode is OUTER_FIT, the function uses the smaller scaling factor, ensuring the whole image
    fits within the target dimensions, potentially leaving some empty space.

    If the resize mode is not OUTER_FIT, the function uses the larger scaling factor, ensuring the target
    dimensions are fully filled, potentially cropping the image.

    After calculating the estimated resolution, the function prints some debugging information.

    Args:
        image (np.ndarray): A 3D numpy array representing an image. The dimensions represent [height, width, channels].
        target_H (int): The target height for the image.
        target_W (int): The target width for the image.
        resize_mode (ResizeMode): The mode for resizing.

    Returns:
        int: The estimated resolution after resizing.
    """
    raw_H, raw_W, _ = image.shape

    k0 = float(target_H) / float(raw_H)
    k1 = float(target_W) / float(raw_W)

    if resize_mode == ResizeMode.OUTER_FIT:
        estimation = min(k0, k1) * float(min(raw_H, raw_W))
    else:
        estimation = max(k0, k1) * float(min(raw_H, raw_W))

    log.debug(f"Pixel Perfect Computation:")
    log.debug(f"resize_mode = {resize_mode}")
    log.debug(f"raw_H = {raw_H}")
    log.debug(f"raw_W = {raw_W}")
    log.debug(f"target_H = {target_H}")
    log.debug(f"target_W = {target_W}")
    log.debug(f"estimation = {estimation}")

    return int(np.round(estimation))

#https://github.com/Mikubill/sd-webui-controlnet/blob/e67e017731aad05796b9615dc6eadce911298ea1/scripts/controlnet.py#L404
def safe_numpy(x):
    # A very safe method to make sure that Apple/Mac works
    y = x

    # below is very boring but do not change these. If you change these Apple or Mac may fail.
    y = y.copy()
    y = np.ascontiguousarray(y)
    y = y.copy()
    return y

#https://github.com/Mikubill/sd-webui-controlnet/blob/e67e017731aad05796b9615dc6eadce911298ea1/scripts/utils.py#L140
def get_unique_axis0(data):
    arr = np.asanyarray(data)
    idxs = np.lexsort(arr.T)
    arr = arr[idxs]
    unique_idxs = np.empty(len(arr), dtype=np.bool_)
    unique_idxs[:1] = True
    unique_idxs[1:] = np.any(arr[:-1, :] != arr[1:, :], axis=-1)
    return arr[unique_idxs]

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

def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z
