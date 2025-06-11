import torch


def batch_rotprojs(batches_rotmats):
    proj_rotmats = []
    for batch_idx, batch_rotmats in enumerate(batches_rotmats):
        proj_batch_rotmats = []
        for rot_idx, rotmat in enumerate(batch_rotmats):
            # GPU implementation of svd is VERY slow
            # ~ 2 10^-3 per hit vs 5 10^-5 on cpu
            U, S, V = rotmat.cpu().svd()
            rotmat = torch.matmul(U, V.transpose(0, 1))
            orth_det = rotmat.det()
            # Remove reflection
            if orth_det < 0:
                rotmat[:, 2] = -1 * rotmat[:, 2]

            rotmat = rotmat.cuda()
            proj_batch_rotmats.append(rotmat)
        proj_rotmats.append(torch.stack(proj_batch_rotmats))
    return torch.stack(proj_rotmats)
