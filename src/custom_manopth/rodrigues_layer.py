"""
This part reuses code from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py
which is part of a PyTorch port of SMPL.
Thanks to Zhang Xiong (MandyMo) for making this great code available on github !
"""

import argparse
from torch.autograd import gradcheck
import torch
from torch.autograd import Variable

from custom_manopth import argutils


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    #axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def th_get_axis_angle(vector):
    angle = torch.norm(vector, 2, 1)
    axes = vector / angle.unsqueeze(1)
    return axes, angle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    argutils.print_args(args)

    n_components = 6
    rot = 3
    inputs = torch.rand(args.batch_size, rot)
    inputs_var = Variable(inputs.double(), requires_grad=True)
    if args.cuda:
        inputs = inputs.cuda()
    # outputs = batch_rodrigues(inputs)
    test_function = gradcheck(batch_rodrigues, (inputs_var, ))
    print('batch test passed !')

    inputs = torch.rand(rot)
    inputs_var = Variable(inputs.double(), requires_grad=True)
    test_function = gradcheck(th_cv2_rod_sub_id.apply, (inputs_var, ))
    print('th_cv2_rod test passed')

    inputs = torch.rand(rot)
    inputs_var = Variable(inputs.double(), requires_grad=True)
    test_th = gradcheck(th_cv2_rod.apply, (inputs_var, ))
    print('th_cv2_rod_id test passed !')
