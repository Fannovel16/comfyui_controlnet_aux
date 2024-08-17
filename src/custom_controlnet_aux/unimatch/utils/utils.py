import torch
import torch.nn.functional as F
import numpy as np


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def bilinear_sampler(img, coords, mode='bilinear', mask=False, padding_mode='zeros'):
    """ Wrapper for grid_sample, uses pixel coordinates """
    if coords.size(-1) != 2:  # [B, 2, H, W] -> [B, H, W, 2]
        coords = coords.permute(0, 2, 3, 1)

    H, W = img.shape[-2:]
    # H = height if height is not None else img.shape[-2]
    # W = width if width is not None else img.shape[-1]

    xgrid, ygrid = coords.split([1, 1], dim=-1)

    # To handle H or W equals to 1 by explicitly defining height and width
    if H == 1:
        assert ygrid.abs().max() < 1e-8
        H = 10
    if W == 1:
        assert xgrid.abs().max() < 1e-8
        W = 10

    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, mode=mode,
                        padding_mode=padding_mode,
                        align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.squeeze(-1).float()

    return img


def coords_grid(batch, ht, wd, normalize=False):
    if normalize:  # [-1, 1]
        coords = torch.meshgrid(2 * torch.arange(ht) / (ht - 1) - 1,
                                2 * torch.arange(wd) / (wd - 1) - 1)
    else:
        coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)  # [B, 2, H, W]


def coords_grid_np(h, w):  # used for accumulating high speed sintel flow testdata
    coords = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing='ij')
    coords = np.stack(coords[::-1], axis=-1)  # [H, W, 2]

    return coords


def compute_out_of_boundary_mask(flow, downsample_factor=None):
    # flow: [B, 2, H, W]
    assert flow.dim() == 4 and flow.size(1) == 2
    b, _, h, w = flow.shape
    init_coords = coords_grid(b, h, w).to(flow.device)
    corres = init_coords + flow  # [B, 2, H, W]

    if downsample_factor is not None:
        assert w % downsample_factor == 0 and h % downsample_factor == 0
        # the actual max disp can predict is in the downsampled feature resolution, then upsample
        max_w = (w // downsample_factor - 1) * downsample_factor
        max_h = (h // downsample_factor - 1) * downsample_factor
        # print('max_w: %d, max_h: %d' % (max_w, max_h))
    else:
        max_w = w - 1
        max_h = h - 1

    valid_mask = (corres[:, 0] >= 0) & (corres[:, 0] <= max_w) & (corres[:, 1] >= 0) & (corres[:, 1] <= max_h)

    # in case very large flow
    flow_mask = (flow[:, 0].abs() <= max_w) & (flow[:, 1].abs() <= max_h)

    valid_mask = valid_mask & flow_mask

    return valid_mask  # [B, H, W]


def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    # grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sampler(feature, grid, mask=mask, padding_mode=padding_mode)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def bilinear_upflow(flow, scale_factor=8):
    assert flow.size(1) == 2
    flow = F.interpolate(flow, scale_factor=scale_factor,
                         mode='bilinear', align_corners=True) * scale_factor

    return flow


def upsample_flow(flow, img):
    if flow.size(-1) != img.size(-1):
        scale_factor = img.size(-1) / flow.size(-1)
        flow = F.interpolate(flow, size=img.size()[-2:],
                             mode='bilinear', align_corners=True) * scale_factor
    return flow


def count_parameters(model):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
