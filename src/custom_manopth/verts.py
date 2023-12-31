'''
Copyright 2017 Javier Romero, Dimitrios Tzionas, Michael J Black and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the MANO/SMPL+H Model license here http://mano.is.tue.mpg.de/license

More information about MANO/SMPL+H is available at http://mano.is.tue.mpg.de.
For comments or questions, please email us at: mano@tue.mpg.de


About this file:
================
This file defines a wrapper for the loading functions of the MANO model.

Modules included:
- load_model:
  loads the MANO model from a given file location (i.e. a .pkl file location),
  or a dictionary object.

'''


import numpy as np
import mano.webuser.lbs as lbs
from mano.webuser.posemapper import posemap
import scipy.sparse as sp


def ischumpy(x):
    return hasattr(x, 'dterms')


def verts_decorated(trans,
                    pose,
                    v_template,
                    J_regressor,
                    weights,
                    kintree_table,
                    bs_style,
                    f,
                    bs_type=None,
                    posedirs=None,
                    betas=None,
                    shapedirs=None,
                    want_Jtr=False):

    for which in [
            trans, pose, v_template, weights, posedirs, betas, shapedirs
    ]:
        if which is not None:
            assert ischumpy(which)

    v = v_template

    if shapedirs is not None:
        if betas is None:
            betas = np.zeros(shapedirs.shape[-1])
        v_shaped = v + shapedirs.dot(betas)
    else:
        v_shaped = v

    if posedirs is not None:
        v_posed = v_shaped + posedirs.dot(posemap(bs_type)(pose))
    else:
        v_posed = v_shaped

    v = v_posed

    if sp.issparse(J_regressor):
        J_tmpx = np.matmul(J_regressor, v_shaped[:, 0])
        J_tmpy = np.matmul(J_regressor, v_shaped[:, 1])
        J_tmpz = np.matmul(J_regressor, v_shaped[:, 2])
        J = np.vstack((J_tmpx, J_tmpy, J_tmpz)).T
    else:
        assert (ischumpy(J))

    assert (bs_style == 'lbs')
    result, Jtr = lbs.verts_core(
        pose, v, J, weights, kintree_table, want_Jtr=True, xp=np)

    tr = trans.reshape((1, 3))
    result = result + tr
    Jtr = Jtr + tr

    result.trans = trans
    result.f = f
    result.pose = pose
    result.v_template = v_template
    result.J = J
    result.J_regressor = J_regressor
    result.weights = weights
    result.kintree_table = kintree_table
    result.bs_style = bs_style
    result.bs_type = bs_type
    if posedirs is not None:
        result.posedirs = posedirs
        result.v_posed = v_posed
    if shapedirs is not None:
        result.shapedirs = shapedirs
        result.betas = betas
        result.v_shaped = v_shaped
    if want_Jtr:
        result.J_transformed = Jtr
    return result


def verts_core(pose,
               v,
               J,
               weights,
               kintree_table,
               bs_style,
               want_Jtr=False,
               xp=np):
    
    assert (bs_style == 'lbs')
    result = lbs.verts_core(pose, v, J, weights, kintree_table, want_Jtr, xp)
    return result