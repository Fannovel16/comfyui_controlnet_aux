import numpy as np
from PIL import Image
from os.path import *
import re
import cv2

TAG_CHAR = np.array([202021.25], np.float32)


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape testdata into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2 ** 15) / 64.0
    return flow, valid


def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    flow = np.stack([-disp, np.zeros_like(disp)], -1)
    return flow, valid


def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2 ** 15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])


def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []


def read_vkitti2_flow(filename):
    # In R, flow along x-axis normalized by image width and quantized to [0;2^16 – 1]
    # In G, flow along x-axis normalized by image width and quantized to [0;2^16 – 1]
    # B = 0 for invalid flow (e.g., sky pixels)
    bgr = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, _c = bgr.shape
    assert bgr.dtype == np.uint16 and _c == 3
    # b == invalid flow flag == 0 for sky or other invalid flow
    invalid = bgr[:, :, 0] == 0
    # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 – 1]
    out_flow = 2.0 / (2 ** 16 - 1.0) * bgr[:, :, 2:0:-1].astype('f4') - 1  # [H, W, 2]
    out_flow[..., 0] *= (w - 1)
    out_flow[..., 1] *= (h - 1)

    out_flow[invalid] = 0.000001  # invalid as very small value to add supervison on the sky
    valid = (np.logical_or(invalid, ~invalid)).astype(np.float32)

    return out_flow, valid
