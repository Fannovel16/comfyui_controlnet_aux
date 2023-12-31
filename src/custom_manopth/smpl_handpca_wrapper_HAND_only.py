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

def col(A):
    return A.reshape((-1, 1))

def MatVecMult(mtx, vec):
    result = mtx.dot(col(vec.ravel())).ravel()
    if len(vec.shape) > 1 and vec.shape[1] > 1:
        result = result.reshape((-1, vec.shape[1]))
    return result

def ready_arguments(fname_or_dict, posekey4vposed='pose'):
    import numpy as np
    import pickle
    from custom_manopth.posemapper import posemap

    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
        # dd = pickle.load(open(fname_or_dict, 'rb'))
    else:
        dd = fname_or_dict

    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1] * 3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in [
            'v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs',
            'betas', 'J'
    ]:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = np.array(dd[s])

    assert (posekey4vposed in dd)
    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas']) + dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:, 0])
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:, 1])
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:, 2])
        dd['J'] = np.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        pose_map_res = posemap(dd['bs_type'])(dd[posekey4vposed])
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(pose_map_res)
    else:
        pose_map_res = posemap(dd['bs_type'])(dd[posekey4vposed])
        dd_add = dd['posedirs'].dot(pose_map_res)
        dd['v_posed'] = dd['v_template'] + dd_add

    return dd


def load_model(fname_or_dict, ncomps=6, flat_hand_mean=False, v_template=None):
    ''' This model loads the fully articulable HAND SMPL model,
    and replaces the pose DOFS by ncomps from PCA'''

    from custom_manopth.verts import verts_core
    import numpy as np
    import pickle
    import scipy.sparse as sp
    np.random.seed(1)

    if not isinstance(fname_or_dict, dict):
        smpl_data = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
        # smpl_data = pickle.load(open(fname_or_dict, 'rb'))
    else:
        smpl_data = fname_or_dict

    rot = 3  # for global orientation!!!

    hands_components = smpl_data['hands_components']
    hands_mean = np.zeros(hands_components.shape[
        1]) if flat_hand_mean else smpl_data['hands_mean']
    hands_coeffs = smpl_data['hands_coeffs'][:, :ncomps]

    selected_components = np.vstack((hands_components[:ncomps]))
    hands_mean = hands_mean.copy()

    pose_coeffs = np.zeros(rot + selected_components.shape[0])
    full_hand_pose = pose_coeffs[rot:(rot + ncomps)].dot(selected_components)

    smpl_data['fullpose'] = np.concatenate((pose_coeffs[:rot],
                                            hands_mean + full_hand_pose))
    smpl_data['pose'] = pose_coeffs

    Jreg = smpl_data['J_regressor']
    if not sp.issparse(Jreg):
        smpl_data['J_regressor'] = (sp.csc_matrix(
            (Jreg.data, (Jreg.row, Jreg.col)), shape=Jreg.shape))

    # slightly modify ready_arguments to make sure that it uses the fullpose
    # (which will NOT be pose) for the computation of posedirs
    dd = ready_arguments(smpl_data, posekey4vposed='fullpose')

    # create the smpl formula with the fullpose,
    # but expose the PCA coefficients as smpl.pose for compatibility
    args = {
        'pose': dd['fullpose'],
        'v': dd['v_posed'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'xp': np,
        'want_Jtr': True,
        'bs_style': dd['bs_style'],
    }

    result_previous, meta = verts_core(**args)

    result = result_previous + dd['trans'].reshape((1, 3))
    result.no_translation = result_previous

    if meta is not None:
        for field in ['Jtr', 'A', 'A_global', 'A_weighted']:
            if (hasattr(meta, field)):
                setattr(result, field, getattr(meta, field))

    setattr(result, 'Jtr', meta)
    if hasattr(result, 'Jtr'):
        result.J_transformed = result.Jtr + dd['trans'].reshape((1, 3))

    for k, v in dd.items():
        setattr(result, k, v)

    if v_template is not None:
        result.v_template[:] = v_template

    return result


if __name__ == '__main__':
    load_model()