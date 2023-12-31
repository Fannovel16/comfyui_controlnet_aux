"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.

Adapted from opensource project GraphCMR (https://github.com/nkolot/GraphCMR/) and Pose2Mesh (https://github.com/hongsukchoi/Pose2Mesh_RELEASE)

"""

from pathlib import Path
folder_path = Path(__file__).parent.parent
JOINT_REGRESSOR_TRAIN_EXTRA = folder_path / 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M_correct = folder_path / 'data/J_regressor_h36m_correct.npy'
SMPL_FILE = folder_path / 'data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
SMPL_Male = folder_path / 'data/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
SMPL_Female = folder_path / 'data/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
SMPL_sampling_matrix = folder_path / 'data/mesh_downsampling.npz'
MANO_FILE = folder_path / 'data/MANO_RIGHT.pkl'
MANO_sampling_matrix = folder_path / 'data/mano_downsampling.npz'

JOINTS_IDX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18, 20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]


"""
We follow the body joint definition, loss functions, and evaluation metrics from 
open source project GraphCMR (https://github.com/nkolot/GraphCMR/)

Each dataset uses different sets of joints.
We use a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are:
"""
J24_NAME = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder',
'L_Elbow','L_Wrist','Neck','Top_of_Head','Pelvis','Thorax','Spine','Jaw','Head','Nose','L_Eye','R_Eye','L_Ear','R_Ear')
H36M_J17_NAME = ( 'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
                  'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
J24_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
H36M_J17_TO_J14 = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 10]

"""
We follow the hand joint definition and mesh topology from 
open source project Manopth (https://github.com/hassony2/manopth)

The hand joints used here are:
"""
J_NAME = ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1',
'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4')
ROOT_INDEX = 0