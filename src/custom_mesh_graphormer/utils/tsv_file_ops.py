"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Basic operations for TSV files
"""


import os
import os.path as op
import json
import numpy as np
import base64
import cv2
from tqdm import tqdm
import yaml
from custom_mesh_graphormer.utils.miscellaneous import mkdir
from custom_mesh_graphormer.utils.tsv_file import TSVFile


def img_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring)
        nparr = np.frombuffer(jpgbytestring, np.uint8)
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return r
    except ValueError:
        return None

def load_linelist_file(linelist_file):
    if linelist_file is not None:
        line_list = []
        with open(linelist_file, 'r') as fp:
            for i in fp:
                line_list.append(int(i.strip()))
        return line_list

def tsv_writer(values, tsv_file, sep='\t'):
    mkdir(op.dirname(tsv_file))
    lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
    idx = 0
    tsv_file_tmp = tsv_file + '.tmp'
    lineidx_file_tmp = lineidx_file + '.tmp'
    with open(tsv_file_tmp, 'w') as fp, open(lineidx_file_tmp, 'w') as fpidx:
        assert values is not None
        for value in values:
            assert value is not None
            value = [v if type(v)!=bytes else v.decode('utf-8') for v in value]
            v = '{0}\n'.format(sep.join(map(str, value)))
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            idx = idx + len(v)
    os.rename(tsv_file_tmp, tsv_file)
    os.rename(lineidx_file_tmp, lineidx_file)

def tsv_reader(tsv_file, sep='\t'):
    with open(tsv_file, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]

def config_save_file(tsv_file, save_file=None, append_str='.new.tsv'):
    if save_file is not None:
        return save_file
    return op.splitext(tsv_file)[0] + append_str

def get_line_list(linelist_file=None, num_rows=None):
    if linelist_file is not None:
        return load_linelist_file(linelist_file)

    if num_rows is not None:
        return [i for i in range(num_rows)]

def generate_hw_file(img_file, save_file=None):
    rows = tsv_reader(img_file)
    def gen_rows():
        for i, row in tqdm(enumerate(rows)):
            row1 = [row[0]]
            img = img_from_base64(row[-1])
            height = img.shape[0]
            width = img.shape[1]
            row1.append(json.dumps([{"height":height, "width": width}]))
            yield row1

    save_file = config_save_file(img_file, save_file, '.hw.tsv')
    tsv_writer(gen_rows(), save_file)

def generate_linelist_file(label_file, save_file=None, ignore_attrs=()):
    # generate a list of image that has labels
    # images with only ignore labels are not selected. 
    line_list = []
    rows = tsv_reader(label_file)
    for i, row in tqdm(enumerate(rows)):
        labels = json.loads(row[1])
        if labels:
            if ignore_attrs and all([any([lab[attr] for attr in ignore_attrs if attr in lab]) \
                                for lab in labels]):
                continue
            line_list.append([i])

    save_file = config_save_file(label_file, save_file, '.linelist.tsv')
    tsv_writer(line_list, save_file)

def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CLoader)

def find_file_path_in_yaml(fname, root):
    if fname is not None:
        if op.isfile(fname):
            return fname
        elif op.isfile(op.join(root, fname)):
            return op.join(root, fname)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), op.join(root, fname)
            )
