from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
from PIL import Image
import sys
import cv2
import json
import os


def read_img(filename):
    # convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
    return img


def read_disp(filename, subset=False, vkitti2=False, sintel=False,
              tartanair=False, instereo2k=False, crestereo=False,
              fallingthings=False,
              argoverse=False,
              raw_disp_png=False,
              ):
    # Scene Flow dataset
    if filename.endswith('pfm'):
        # For finalpass and cleanpass, gt disparity is positive, subset is negative
        disp = np.ascontiguousarray(_read_pfm(filename)[0])
        if subset:
            disp = -disp
    # VKITTI2 dataset
    elif vkitti2:
        disp = _read_vkitti2_disp(filename)
    # Sintel
    elif sintel:
        disp = _read_sintel_disparity(filename)
    elif tartanair:
        disp = _read_tartanair_disp(filename)
    elif instereo2k:
        disp = _read_instereo2k_disp(filename)
    elif crestereo:
        disp = _read_crestereo_disp(filename)
    elif fallingthings:
        disp = _read_fallingthings_disp(filename)
    elif argoverse:
        disp = _read_argoverse_disp(filename)
    elif raw_disp_png:
        disp = np.array(Image.open(filename)).astype(np.float32)
    # KITTI
    elif filename.endswith('png'):
        disp = _read_kitti_disp(filename)
    elif filename.endswith('npy'):
        disp = np.load(filename)
    else:
        raise Exception('Invalid disparity file format!')
    return disp  # [H, W]


def _read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(
            image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image.tofile(file)


def _read_kitti_disp(filename):
    depth = np.array(Image.open(filename))
    depth = depth.astype(np.float32) / 256.
    return depth


def _read_vkitti2_disp(filename):
    # read depth
    depth = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # in cm
    depth = (depth / 100).astype(np.float32)  # depth clipped to 655.35m for sky

    valid = (depth > 0) & (depth < 655)  # depth clipped to 655.35m for sky

    # convert to disparity
    focal_length = 725.0087  # in pixels
    baseline = 0.532725  # meter

    disp = baseline * focal_length / depth

    disp[~valid] = 0.000001  # invalid as very small value

    return disp


def _read_sintel_disparity(filename):
    """ Return disparity read from filename. """
    f_in = np.array(Image.open(filename))

    d_r = f_in[:, :, 0].astype('float32')
    d_g = f_in[:, :, 1].astype('float32')
    d_b = f_in[:, :, 2].astype('float32')

    depth = d_r * 4 + d_g / (2 ** 6) + d_b / (2 ** 14)
    return depth


def _read_tartanair_disp(filename):
    # the infinite distant object such as the sky has a large depth value (e.g. 10000)
    depth = np.load(filename)

    # change to disparity image
    disparity = 80.0 / depth

    return disparity


def _read_instereo2k_disp(filename):
    disp = np.array(Image.open(filename))
    disp = disp.astype(np.float32) / 100.
    return disp


def _read_crestereo_disp(filename):
    disp = np.array(Image.open(filename))
    return disp.astype(np.float32) / 32.


def _read_fallingthings_disp(filename):
    depth = np.array(Image.open(filename))
    camera_file = os.path.join(os.path.dirname(filename), '_camera_settings.json')
    with open(camera_file, 'r') as f:
        intrinsics = json.load(f)
    fx = intrinsics['camera_settings'][0]['intrinsic_settings']['fx']
    disp = (fx * 6.0 * 100) / depth.astype(np.float32)

    return disp


def _read_argoverse_disp(filename):
    disparity_map = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return np.float32(disparity_map) / 256.


def extract_video(video_name):
    cap = cv2.VideoCapture(video_name)
    assert cap.isOpened(), f'Failed to load video file {video_name}'
    # get video info
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('video size (hxw): %dx%d' % (size[1], size[0]))
    print('fps: %d' % fps)

    imgs = []
    while cap.isOpened():
        # get frames
        flag, img = cap.read()
        if not flag:
            break
        # to rgb format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    return imgs, fps
