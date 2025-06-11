import glob
import os
import json
import cv2

def load_from_annos(anno_path):
    with open(anno_path, 'r') as f:
        annos = json.load(f)['files']

    datas = []
    for i, anno in enumerate(annos):
        rgb = anno['rgb']
        depth = anno['depth'] if 'depth' in anno else None
        depth_scale = anno['depth_scale'] if 'depth_scale' in anno else 1.0
        intrinsic = anno['cam_in'] if 'cam_in' in anno else None
        normal = anno['normal'] if 'normal' in anno else None

        data_i = {
            'rgb': rgb,
            'depth': depth,
            'depth_scale': depth_scale,
            'intrinsic': intrinsic,
            'filename': os.path.basename(rgb),
            'folder': rgb.split('/')[-3],
            'normal': normal
        }
        datas.append(data_i)
    return datas

def load_data(path: str):
    rgbs = glob.glob(path + '/*.jpg') + glob.glob(path + '/*.png')
    #intrinsic =  [835.8179931640625, 835.8179931640625, 961.5419921875, 566.8090209960938] #[721.53769, 721.53769, 609.5593, 172.854]
    data = [{'rgb': i, 'depth': None, 'intrinsic': None, 'filename': os.path.basename(i), 'folder': i.split('/')[-3]} for i in rgbs]
    return data