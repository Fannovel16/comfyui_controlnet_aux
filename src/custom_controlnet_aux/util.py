import os
import random
import tempfile
import warnings
from contextlib import suppress
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import constants, hf_hub_download
from torch.utils.model_zoo import load_url
from ast import literal_eval


HF_MODEL_NAME = "lllyasviel/Annotators"
DWPOSE_MODEL_NAME = "yzd-v/DWPose"
BDS_MODEL_NAME = "bdsqlsz/qinglong_controlnet-lllite"
DENSEPOSE_MODEL_NAME = "LayerNorm/DensePose-TorchScript-with-hint-image"
MESH_GRAPHORMER_MODEL_NAME = "hr16/ControlNet-HandRefiner-pruned"
SAM_MODEL_NAME = "dhkim2810/MobileSAM"
UNIMATCH_MODEL_NAME = "hr16/Unimatch"
DEPTH_ANYTHING_MODEL_NAME = "LiheYoung/Depth-Anything" #HF Space
DIFFUSION_EDGE_MODEL_NAME = "hr16/Diffusion-Edge"
METRIC3D_MODEL_NAME = "JUGGHM/Metric3D"

DEPTH_ANYTHING_V2_MODEL_NAME_DICT = {
    "depth_anything_v2_vits.pth": "depth-anything/Depth-Anything-V2-Small",
    "depth_anything_v2_vitb.pth": "depth-anything/Depth-Anything-V2-Base",
    "depth_anything_v2_vitl.pth": "depth-anything/Depth-Anything-V2-Large",
    "depth_anything_v2_vitg.pth": "depth-anything/Depth-Anything-V2-Giant",
    "depth_anything_v2_metric_vkitti_vitl.pth": "depth-anything/Depth-Anything-V2-Metric-VKITTI-Large",
    "depth_anything_v2_metric_hypersim_vitl.pth": "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large"
}

temp_dir = tempfile.gettempdir()
annotator_ckpts_path = os.path.join(Path(__file__).parents[2], 'ckpts')
USE_SYMLINKS = False

try:
    annotator_ckpts_path = os.environ['AUX_ANNOTATOR_CKPTS_PATH']
except:
    warnings.warn("Custom pressesor model path not set successfully.")
    pass

try:
    USE_SYMLINKS = literal_eval(os.environ['AUX_USE_SYMLINKS'])
except:
    warnings.warn("USE_SYMLINKS not set successfully. Using default value: False to download models.")
    pass

try:
    temp_dir = os.environ['AUX_TEMP_DIR']
    if len(temp_dir) >= 60:
        warnings.warn(f"custom temp dir is too long. Using default")
        temp_dir = tempfile.gettempdir()
except:
    warnings.warn(f"custom temp dir not set successfully")
    pass

here = Path(__file__).parent.resolve()

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def make_noise_disk(H, W, C, F, rng=None):
    if rng:
        noise = rng.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    else:
        noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F: F + H, F: F + W]
    noise -= np.min(noise)
    noise /= np.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise


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

def min_max_norm(x):
    x -= np.min(x)
    x /= np.maximum(np.max(x), 1e-5)
    return x


def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


def img2mask(img, H, W, low=10, high=90):
    assert img.ndim == 3 or img.ndim == 2
    assert img.dtype == np.uint8

    if img.ndim == 3:
        y = img[:, :, random.randrange(0, img.shape[2])]
    else:
        y = img

    y = cv2.resize(y, (W, H), interpolation=cv2.INTER_CUBIC)

    if random.uniform(0, 1) < 0.5:
        y = 255 - y

    return y < np.percentile(y, random.randrange(low, high))

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

UPSCALE_METHODS = ["INTER_NEAREST", "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"]
def get_upscale_method(method_str):
    assert method_str in UPSCALE_METHODS, f"Method {method_str} not found in {UPSCALE_METHODS}"
    return getattr(cv2, method_str)

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

#https://github.com/Mikubill/sd-webui-controlnet/blob/main/scripts/processor.py#L17
#Added upscale_method, mode params
def resize_image_with_pad(input_image, resolution, upscale_method = "", skip_hwc3=False, mode='edge'):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    if resolution == 0:
        return img, lambda x: x
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=get_upscale_method(upscale_method) if k > 1 else cv2.INTER_AREA)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target, ...])

    return safer_memory(img_padded), remove_pad
    
def common_input_validate(input_image, output_type, **kwargs):
    if "img" in kwargs:
            warnings.warn("img is deprecated, please use `input_image=...` instead.", DeprecationWarning)
            input_image = kwargs.pop("img")
    
    if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
    
    if type(output_type) is bool:
        warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
        if output_type:
            output_type = "pil"

    if input_image is None:
        raise ValueError("input_image must be defined.")

    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image, dtype=np.uint8)
        output_type = output_type or "pil"
    else:
        output_type = output_type or "np"
    
    return (input_image, output_type)

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

#https://stackoverflow.com/a/44873382
#Assume that the minimum version of Python ppl use is 3.9
def sha256sum(file_path):
    import hashlib
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(file_path, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()

def check_hash_from_torch_hub(file_path, filename):
    basename, _ = filename.split('.')
    _, ref_hash = basename.split('-')
    curr_hash = sha256sum(file_path)
    return curr_hash[:len(ref_hash)] == ref_hash

def custom_torch_download(filename, ckpts_dir=annotator_ckpts_path):
    """Download PyTorch models using PyTorch 2.7's built-in download mechanism."""
    model_url = "https://download.pytorch.org/models/" + filename
    
    # Use PyTorch's built-in model downloading with custom cache directory
    local_dir = os.path.join(ckpts_dir, "torch")
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
    
    model_path = os.path.join(local_dir, filename)
    
    if not os.path.exists(model_path):
        print(f"Downloading {filename} from pytorch.org...")
        try:
            # Use PyTorch 2.7's load_url which handles caching, progress, and hash checking
            state_dict = load_url(model_url, model_dir=local_dir, file_name=filename, progress=True, check_hash=True)
            # The file is already saved by load_url, we just need the path
        except Exception as e:
            warnings.warn(f"Download failed with error: {e}")
            raise
    
    print(f"model_path is {model_path}")
    return model_path

def custom_hf_download(pretrained_model_or_path, filename, cache_dir=temp_dir, ckpts_dir=annotator_ckpts_path, subfolder='', use_symlinks=USE_SYMLINKS, repo_type="model"):

    local_dir = os.path.join(ckpts_dir, pretrained_model_or_path)
    model_path = Path(local_dir).joinpath(*subfolder.split('/'), filename).__str__()

    if len(str(model_path)) >= 255:
        warnings.warn(f"Path {model_path} is too long, \n please change annotator_ckpts_path in config.yaml")

    if not os.path.exists(model_path):
        print(f"Failed to find {model_path}.\n Downloading from huggingface.co")
        print(f"cacher folder is {cache_dir}, you can change it by custom_tmp_path in config.yaml")
        if use_symlinks:
            cache_dir_d = constants.HF_HUB_CACHE    # use huggingface newer env variables `HF_HUB_CACHE`
            if cache_dir_d is None:
                import platform
                if platform.system() == "Windows":
                    cache_dir_d = Path(os.getenv("USERPROFILE")).joinpath(".cache", "huggingface", "hub").__str__()
                else:
                    cache_dir_d = os.path.join(os.getenv("HOME"), ".cache", "huggingface", "hub")
            try:
                # test_link
                Path(cache_dir_d).mkdir(parents=True, exist_ok=True)
                Path(ckpts_dir).mkdir(parents=True, exist_ok=True)
                (Path(cache_dir_d) / f"linktest_{filename}.txt").touch()
                # symlink instead of link avoid `invalid cross-device link` error.
                os.symlink(os.path.join(cache_dir_d, f"linktest_{filename}.txt"), os.path.join(ckpts_dir, f"linktest_{filename}.txt"))
                print("Using symlinks to download models. \n",\
                      "Make sure you have enough space on your cache folder. \n",\
                      "And do not purge the cache folder after downloading.\n",\
                      "Otherwise, you will have to re-download the models every time you run the script.\n",\
                      "You can use USE_SYMLINKS: False in config.yaml to avoid this behavior.")
            except:
                print("Maybe not able to create symlink. Disable using symlinks.")
                use_symlinks = False
                cache_dir_d = Path(cache_dir).joinpath("ckpts", pretrained_model_or_path).__str__()
            finally:    # always remove test link files
                with suppress(FileNotFoundError):
                    os.remove(os.path.join(ckpts_dir, f"linktest_{filename}.txt"))
                    os.remove(os.path.join(cache_dir_d, f"linktest_{filename}.txt"))
        else:
            cache_dir_d = os.path.join(cache_dir, "ckpts", pretrained_model_or_path)

        model_path = hf_hub_download(repo_id=pretrained_model_or_path,
            cache_dir=cache_dir_d,
            local_dir=local_dir,
            subfolder=subfolder,
            filename=filename,
            local_dir_use_symlinks=use_symlinks,
            resume_download=True,
            etag_timeout=100,
            repo_type=repo_type
        )
        if not use_symlinks:
            try:
                import shutil
                shutil.rmtree(os.path.join(cache_dir, "ckpts"))
            except Exception as e :
                print(e)

    print(f"model_path is {model_path}")

    return model_path
