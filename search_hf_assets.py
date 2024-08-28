from pathlib import Path
import os
import re
#Thanks ChatGPT
pattern = r'\bfrom_pretrained\(.*?pretrained_model_or_path\s*=\s*(.*?)(?:,|\))|filename\s*=\s*(.*?)(?:,|\))|(\w+_filename)\s*=\s*(.*?)(?:,|\))'
aux_dir = Path(__file__).parent / 'src' / 'custom_controlnet_aux'
VAR_DICT = dict(
    HF_MODEL_NAME = "lllyasviel/Annotators",
    DWPOSE_MODEL_NAME = "yzd-v/DWPose",
    BDS_MODEL_NAME = "bdsqlsz/qinglong_controlnet-lllite",
    DENSEPOSE_MODEL_NAME = "LayerNorm/DensePose-TorchScript-with-hint-image",
    MESH_GRAPHORMER_MODEL_NAME = "hr16/ControlNet-HandRefiner-pruned",
    SAM_MODEL_NAME = "dhkim2810/MobileSAM",
    UNIMATCH_MODEL_NAME = "hr16/Unimatch",
    DEPTH_ANYTHING_MODEL_NAME = "LiheYoung/Depth-Anything", #HF Space
    DIFFUSION_EDGE_MODEL_NAME = "hr16/Diffusion-Edge"
)
re_result_dict = {}
for preprocc in os.listdir(aux_dir):
    if preprocc in ["__pycache__", 'tests']: continue
    if '.py' in preprocc: continue
    f = open(aux_dir / preprocc / '__init__.py', 'r')
    code = f.read()
    matches = re.findall(pattern, code)
    result = [match[0] or match[1] or match[3] for match in matches]
    if not len(result):
        print(preprocc)
        continue
    result = [el.replace("'", '').replace('"', '') for el in result]
    result = [VAR_DICT.get(el, el) for el in result]
    re_result_dict[preprocc] = result
    f.close()

for preprocc, re_result in re_result_dict.items():
    model_name, filenames = re_result[0], re_result[1:]
    print(f"* {preprocc}: ", end=' ')
    assests_md = ', '.join([f"[{model_name}/{filename}](https://huggingface.co/{model_name}/blob/main/{filename})" for filename in filenames])
    print(assests_md)

preprocc = "dwpose"
model_name, filenames = VAR_DICT['DWPOSE_MODEL_NAME'], ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"]
print(f"* {preprocc}: ", end=' ')
assests_md = ', '.join([f"[{model_name}/{filename}](https://huggingface.co/{model_name}/blob/main/{filename})" for filename in filenames])
print(assests_md)

preprocc = "yolo-nas"
model_name, filenames = "hr16/yolo-nas-fp16", ["yolo_nas_l_fp16.onnx", "yolo_nas_m_fp16.onnx", "yolo_nas_s_fp16.onnx"]
print(f"* {preprocc}: ", end=' ')
assests_md = ', '.join([f"[{model_name}/{filename}](https://huggingface.co/{model_name}/blob/main/{filename})" for filename in filenames])
print(assests_md)

preprocc = "dwpose-torchscript"
model_name, filenames = "hr16/DWPose-TorchScript-BatchSize5", ["dw-ll_ucoco_384_bs5.torchscript.pt", "rtmpose-m_ap10k_256_bs5.torchscript.pt"]
print(f"* {preprocc}: ", end=' ')
assests_md = ', '.join([f"[{model_name}/{filename}](https://huggingface.co/{model_name}/blob/main/{filename})" for filename in filenames])
print(assests_md)