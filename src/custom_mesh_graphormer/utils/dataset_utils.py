"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""


import os
import os.path as op
import numpy as np
import base64
import cv2
import yaml
from collections import OrderedDict


def img_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring)
        nparr = np.frombuffer(jpgbytestring, np.uint8)
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return r
    except:
        return None


def load_labelmap(labelmap_file):
    label_dict = None
    if labelmap_file is not None and op.isfile(labelmap_file):
        label_dict = OrderedDict()
        with open(labelmap_file, 'r') as fp:
            for line in fp:
                label = line.strip().split('\t')[0]
                if label in label_dict:
                    raise ValueError("Duplicate label " + label + " in labelmap.")
                else:
                    label_dict[label] = len(label_dict)
    return label_dict


def load_shuffle_file(shuf_file):
    shuf_list = None
    if shuf_file is not None:
        with open(shuf_file, 'r') as fp:
            shuf_list = []
            for i in fp:
                shuf_list.append(int(i.strip()))
    return shuf_list


def load_box_shuffle_file(shuf_file):
    if shuf_file is not None:
        with open(shuf_file, 'r') as fp:
            img_shuf_list = []
            box_shuf_list = []
            for i in fp:
                idx = [int(_) for _ in i.strip().split('\t')]
                img_shuf_list.append(idx[0])
                box_shuf_list.append(idx[1])
        return [img_shuf_list, box_shuf_list]
    return None


def load_from_yaml_file(file_name):
    with open(file_name, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CLoader)
