"""
This file contains the definition of the SMPL model

It is adapted from opensource project GraphCMR (https://github.com/nkolot/GraphCMR/)
"""
from __future__ import division

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
try:
    import cPickle as pickle
except ImportError:
    import pickle

from custom_mesh_graphormer.utils.geometric_layers import rodrigues
import custom_mesh_graphormer.modeling.data.config as cfg

from comfy.model_management import get_torch_device
from wrapper_for_mps import sparse_to_dense
device = get_torch_device()

class SMPL(nn.Module):

    def __init__(self, gender='neutral'):
        super(SMPL, self).__init__()

        if gender=='m':
            model_file=cfg.SMPL_Male
        elif gender=='f':
            model_file=cfg.SMPL_Female
        else:
            model_file=cfg.SMPL_FILE

        smpl_model = pickle.load(open(model_file, 'rb'), encoding='latin1') 
        J_regressor = smpl_model['J_regressor'].tocoo()
        row = J_regressor.row
        col = J_regressor.col
        data = J_regressor.data
        i = torch.LongTensor([row, col])
        v = torch.FloatTensor(data)
        J_regressor_shape = [24, 6890]
        self.register_buffer('J_regressor', torch.sparse_coo_tensor(i, v, J_regressor_shape).to_dense())
        self.register_buffer('weights', torch.FloatTensor(smpl_model['weights']))
        self.register_buffer('posedirs', torch.FloatTensor(smpl_model['posedirs']))
        self.register_buffer('v_template', torch.FloatTensor(smpl_model['v_template']))
        self.register_buffer('shapedirs', torch.FloatTensor(np.array(smpl_model['shapedirs'])))
        self.register_buffer('faces', torch.from_numpy(smpl_model['f'].astype(np.int64)))
        self.register_buffer('kintree_table', torch.from_numpy(smpl_model['kintree_table'].astype(np.int64)))
        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.register_buffer('parent', torch.LongTensor([id_to_col[self.kintree_table[0, it].item()] for it in range(1, self.kintree_table.shape[1])]))

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.translation_shape = [3]

        self.pose = torch.zeros(self.pose_shape)
        self.beta = torch.zeros(self.beta_shape)
        self.translation = torch.zeros(self.translation_shape)

        self.verts = None
        self.J = None
        self.R = None
        
        J_regressor_extra = torch.from_numpy(np.load(cfg.JOINT_REGRESSOR_TRAIN_EXTRA)).float()
        self.register_buffer('J_regressor_extra', J_regressor_extra)
        self.joints_idx = cfg.JOINTS_IDX

        J_regressor_h36m_correct = torch.from_numpy(np.load(cfg.JOINT_REGRESSOR_H36M_correct)).float()
        self.register_buffer('J_regressor_h36m_correct', J_regressor_h36m_correct)


    def forward(self, pose, beta):
        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1,10)[None, :].expand(batch_size, -1, -1)
        beta = beta[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template
        # batched sparse matmul not supported in pytorch
        J = []
        for i in range(batch_size):
            J.append(torch.matmul(self.J_regressor, v_shaped[i]))
        J = torch.stack(J, dim=0)
        # input it rotmat: (bs,24,3,3)
        if pose.ndimension() == 4:
            R = pose
        # input it rotmat: (bs,72)
        elif pose.ndimension() == 2:
            pose_cube = pose.view(-1, 3) # (batch_size * 24, 1, 3)
            R = rodrigues(pose_cube).view(batch_size, 24, 3, 3)
            R = R.view(batch_size, 24, 3, 3)
        I_cube = torch.eye(3)[None, None, :].to(device)
        # I_cube = torch.eye(3)[None, None, :].expand(theta.shape[0], R.shape[1]-1, -1, -1)
        lrotmin = (R[:,1:,:] - I_cube).view(batch_size, -1)
        posedirs = self.posedirs.view(-1,207)[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(posedirs, lrotmin[:, :, None]).view(-1, 6890, 3)
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = torch.FloatTensor([0,0,0,1]).to(device).view(1,1,1,4).expand(batch_size, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i-1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(batch_size, 24, 4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(self.weights, G.permute(1,0,2,3).contiguous().view(24,-1)).view(6890, batch_size, 4, 4).transpose(0,1)
        rest_shape_h = torch.cat([v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        return v

    def get_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 38, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor])
        joints_extra = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor_extra])
        joints = torch.cat((joints, joints_extra), dim=1)
        joints = joints[:, cfg.JOINTS_IDX]
        return joints

    def get_h36m_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor_h36m_correct])
        return joints

class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input

def spmm(sparse, dense):
    sparse = sparse.to(device)
    dense = dense.to(device)
    return SparseMM.apply(sparse, dense)


def scipy_to_pytorch(A, U, D):
    """Convert scipy sparse matrices to pytorch sparse matrix."""
    ptU = []
    ptD = []
    
    for i in range(len(U)):
        u = scipy.sparse.coo_matrix(U[i])
        i = torch.LongTensor(np.array([u.row, u.col]))
        v = torch.FloatTensor(u.data)
        ptU.append(sparse_to_dense(torch.sparse_coo_tensor(i, v, u.shape)))
    
    for i in range(len(D)):
        d = scipy.sparse.coo_matrix(D[i])
        i = torch.LongTensor(np.array([d.row, d.col]))
        v = torch.FloatTensor(d.data)
        ptD.append(sparse_to_dense(torch.sparse_coo_tensor(i, v, d.shape))) 

    return ptU, ptD


def adjmat_sparse(adjmat, nsize=1):
    """Create row-normalized sparse graph adjacency matrix."""
    adjmat = scipy.sparse.csr_matrix(adjmat)
    if nsize > 1:
        orig_adjmat = adjmat.copy()
        for _ in range(1, nsize):
            adjmat = adjmat * orig_adjmat
    adjmat.data = np.ones_like(adjmat.data)
    for i in range(adjmat.shape[0]):
        adjmat[i,i] = 1
    num_neighbors = np.array(1 / adjmat.sum(axis=-1))
    adjmat = adjmat.multiply(num_neighbors)
    adjmat = scipy.sparse.coo_matrix(adjmat)
    row = adjmat.row
    col = adjmat.col
    data = adjmat.data
    i = torch.LongTensor(np.array([row, col]))
    v = torch.from_numpy(data).float()
    adjmat = sparse_to_dense(torch.sparse_coo_tensor(i, v, adjmat.shape))
    return adjmat

def get_graph_params(filename, nsize=1):
    """Load and process graph adjacency matrix and upsampling/downsampling matrices."""
    data = np.load(filename, encoding='latin1', allow_pickle=True)
    A = data['A']
    U = data['U']
    D = data['D']
    U, D = scipy_to_pytorch(A, U, D)
    A = [adjmat_sparse(a, nsize=nsize) for a in A]
    return A, U, D


class Mesh(object):
    """Mesh object that is used for handling certain graph operations."""
    def __init__(self, filename=cfg.SMPL_sampling_matrix,
                 num_downsampling=1, nsize=1, device=torch.device('cuda')):
        self._A, self._U, self._D = get_graph_params(filename=filename, nsize=nsize)
        # self._A = [a.to(device) for a in self._A]
        self._U = [u.to(device) for u in self._U]
        self._D = [d.to(device) for d in self._D]
        self.num_downsampling = num_downsampling

        # load template vertices from SMPL and normalize them
        smpl = SMPL()
        ref_vertices = smpl.v_template
        center = 0.5*(ref_vertices.max(dim=0)[0] + ref_vertices.min(dim=0)[0])[None]
        ref_vertices -= center
        ref_vertices /= ref_vertices.abs().max().item()

        self._ref_vertices = ref_vertices.to(device)
        self.faces = smpl.faces.int().to(device)

    # @property
    # def adjmat(self):
    #     """Return the graph adjacency matrix at the specified subsampling level."""
    #     return self._A[self.num_downsampling].float()

    @property
    def ref_vertices(self):
        """Return the template vertices at the specified subsampling level."""
        ref_vertices = self._ref_vertices
        for i in range(self.num_downsampling):
            ref_vertices = torch.spmm(self._D[i], ref_vertices)
        return ref_vertices

    def downsample(self, x, n1=0, n2=None):
        """Downsample mesh."""
        if n2 is None:
            n2 = self.num_downsampling
        if x.ndimension() < 3:
            for i in range(n1, n2):
                x = spmm(self._D[i], x)
        elif x.ndimension() == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in range(n1, n2):
                    y = spmm(self._D[j], y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x

    def upsample(self, x, n1=1, n2=0):
        """Upsample mesh."""
        if x.ndimension() < 3:
            for i in reversed(range(n2, n1)):
                x = spmm(self._U[i], x)
        elif x.ndimension() == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in reversed(range(n2, n1)):
                    y = spmm(self._U[j], y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x
