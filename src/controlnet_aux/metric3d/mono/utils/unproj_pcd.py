import numpy as np
import torch
from plyfile import PlyData, PlyElement
import cv2


def get_pcd_base(H, W, u0, v0, fx, fy):
    x_row = np.arange(0, W)
    x = np.tile(x_row, (H, 1))
    x = x.astype(np.float32)
    u_m_u0 = x - u0

    y_col = np.arange(0, H)  # y_col = np.arange(0, height)
    y = np.tile(y_col, (W, 1)).T
    y = y.astype(np.float32)
    v_m_v0 = y - v0

    x = u_m_u0 / fx
    y = v_m_v0 / fy
    z = np.ones_like(x)
    pw = np.stack([x, y, z], axis=2)  # [h, w, c]
    return pw


def reconstruct_pcd(depth, fx, fy, u0, v0, pcd_base=None, mask=None):
    if type(depth) == torch.__name__:
        depth = depth.cpu().numpy().squeeze()
    depth = cv2.medianBlur(depth, 5)
    if pcd_base is None:
        H, W = depth.shape
        pcd_base = get_pcd_base(H, W, u0, v0, fx, fy)
    pcd = depth[:, :, None] * pcd_base
    if mask:
        pcd[mask] = 0
    return pcd


def save_point_cloud(pcd, rgb, filename, binary=True):
    """Save an RGB point cloud as a PLY file.
    :paras
        @pcd: Nx3 matrix, the XYZ coordinates
        @rgb: Nx3 matrix, the rgb colors for each 3D point
    """
    assert pcd.shape[0] == rgb.shape[0]

    if rgb is None:
        gray_concat = np.tile(np.array([128], dtype=np.uint8),
                              (pcd.shape[0], 3))
        points_3d = np.hstack((pcd, gray_concat))
    else:
        points_3d = np.hstack((pcd, rgb))
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                 ('green', 'u1'), ('blue', 'u1')]
    if binary is True:
        # Format into Numpy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(
                tuple(
                    dtype(point)
                    for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

        # write
        PlyData([el]).write(filename)
    else:
        x = np.squeeze(points_3d[:, 0])
        y = np.squeeze(points_3d[:, 1])
        z = np.squeeze(points_3d[:, 2])
        r = np.squeeze(points_3d[:, 3])
        g = np.squeeze(points_3d[:, 4])
        b = np.squeeze(points_3d[:, 5])

        ply_head = 'ply\n' \
                    'format ascii 1.0\n' \
                    'element vertex %d\n' \
                    'property float x\n' \
                    'property float y\n' \
                    'property float z\n' \
                    'property uchar red\n' \
                    'property uchar green\n' \
                    'property uchar blue\n' \
                    'end_header' % r.shape[0]
        # ---- Save ply data to disk
        np.savetxt(filename, np.column_stack[x, y, z, r, g, b], fmt='%f %f %f %d %d %d', header=ply_head, comments='')