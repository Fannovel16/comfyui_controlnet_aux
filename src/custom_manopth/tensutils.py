import torch

from custom_manopth import rodrigues_layer


def th_posemap_axisang(pose_vectors):
    rot_nb = int(pose_vectors.shape[1] / 3)
    pose_vec_reshaped = pose_vectors.contiguous().view(-1, 3)
    rot_mats = rodrigues_layer.batch_rodrigues(pose_vec_reshaped)
    rot_mats = rot_mats.view(pose_vectors.shape[0], rot_nb * 9)
    pose_maps = subtract_flat_id(rot_mats)
    return pose_maps, rot_mats


def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False

    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res


def th_pack(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new_zeros((batch_size, 4, 3))
    padding.requires_grad = False
    pack_list = [padding, tensor]
    pack_res = torch.cat(pack_list, 2)
    return pack_res


def subtract_flat_id(rot_mats):
    # Subtracts identity as a flattened tensor
    rot_nb = int(rot_mats.shape[1] / 9)
    id_flat = torch.eye(
        3, dtype=rot_mats.dtype, device=rot_mats.device).view(1, 9).repeat(
            rot_mats.shape[0], rot_nb)
    # id_flat.requires_grad = False
    results = rot_mats - id_flat
    return results


def make_list(tensor):
    # type: (List[int]) -> List[int]
    return tensor
