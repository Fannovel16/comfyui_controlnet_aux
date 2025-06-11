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
import cv2

def lrotmin(p):
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate(
            [(cv2.Rodrigues(np.array(pp))[0] - np.eye(3)).ravel()
             for pp in p.reshape((-1, 3))]).ravel()


def posemap(s):
    if s == 'lrotmin':
        return lrotmin
    else:
        raise Exception('Unknown posemapping: %s' % (str(s), ))