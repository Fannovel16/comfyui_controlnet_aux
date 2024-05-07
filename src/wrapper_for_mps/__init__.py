import torch
from comfy.model_management import get_torch_device

device = get_torch_device()
#https://github.com/microsoft/DirectML/issues/414#issuecomment-1541319479
def sparse_to_dense(sparse_tensor):
    return sparse_tensor.to_dense()