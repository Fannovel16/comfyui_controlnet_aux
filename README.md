# ComfyUI's ControlNet Preprocessors auxiliary models

This is a WIP rework of [comfyui_controlnet_preprocessors](https://github.com/Fannovel16/comfy_controlnet_preprocessors) based on [ControlNet auxiliary models by 🤗](https://github.com/patrickvonplaten/controlnet_aux). I think the old repo isn't good enough to maintain.

All old workflow will still be work with this repo but the version option won't do anything. Almost all v1 preprocessors are replaced by v1.1 except those doesn't apppear in v1.1.

The code is copy-pasted from the respective folders in https://github.com/lllyasviel/ControlNet/tree/main/annotator and connected to [the 🤗 Hub](https://huggingface.co/lllyasviel/Annotators).

All credit & copyright goes to https://github.com/lllyasviel.

# Installation:
## Using ComfyUI Manager (recommended):
Install [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) and do steps introduced there to install this repo.

## Alternative:
If you're running on Linux, or non-admin account on windows you'll want to ensure `/ComfyUI/custom_nodes` and `comfyui_controlnet_aux` has write permissions.

There is now a **install.bat** you can run to install to portable if detected. Otherwise it will default to system and assume you followed ConfyUI's manual installation steps. 

If you can't run **install.bat** (e.g. you are a Linux user). Open the CMD/Shell and do the following:
  - Navigate to your `/ComfyUI/custom_nodes/` folder
  - Run `git clone https://github.com/Fannovel16/comfyui_controlnet_aux/`
  - Navigate to your `was-node-suite-comfyui` folder
    - Portable/venv:
       - Run `path/to/ComfUI/python_embeded/python.exe -s -m pip install -r requirements.txt`
	- With system python
	   - Run `pip install -r requirements.txt`
  - Start ComfyUI

## Nodes
- HEDPreprocessor
- ScribblePreprocessor
- FakeScribblePreprocessor
- PiDiNetPreprocessor
- LineArtPreprocessor
- AnimeLineArtPreprocessor
- Manga2Anime-LineArtPreprocessor
- CannyEdgePreprocessor
- M-LSDPreprocessor
- DWPosePreprocessor

## Testing workflow
If you meet OOM, just click Queue Prompt one more time and it will continue from the unfinished node.
https://github.com/Fannovel16/comfyui_controlnet_aux/blob/master/tests/test_cn_aux_workflow.json
![](https://github.com/Fannovel16/comfyui_controlnet_aux/blob/master/tests/pose.png?raw=true)
## Developer Interface
Since this repo is mainly used for ComfyUI, you can't import models directly like the original ControlNet Aux.

Use `controlnet_aux.dev_interface` instead of `controlnet_aux`.

```py
from PIL import Image
import requests
from io import BytesIO
#Here
from controlnet_aux.dev_interface import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector

# load image
url = "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png"

response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))

# load checkpoints
hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators")
open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
normal_bae = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
sam = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
mobile_sam = SamDetector.from_pretrained("dhkim2810/MobileSAM", model_type="vit_t", filename="mobile_sam.pt")
leres = LeresDetector.from_pretrained("lllyasviel/Annotators")

# instantiate
canny = CannyDetector()
content = ContentShuffleDetector()
face_detector = MediapipeFaceDetector()


# process
processed_image_hed = hed(img)
processed_image_midas = midas(img)
processed_image_mlsd = mlsd(img)
processed_image_open_pose = open_pose(img, hand_and_face=True)
processed_image_pidi = pidi(img, safe=True)
processed_image_normal_bae = normal_bae(img)
processed_image_lineart = lineart(img, coarse=True)
processed_image_lineart_anime = lineart_anime(img)
processed_image_zoe = zoe(img)
processed_image_sam = sam(img)
processed_image_leres = leres(img)

processed_image_canny = canny(img)
processed_image_content = content(img)
processed_image_mediapipe_face = face_detector(img)
```