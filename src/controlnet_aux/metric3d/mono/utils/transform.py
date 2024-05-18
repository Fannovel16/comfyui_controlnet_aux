import collections
import cv2
import math
import numpy as np
import numbers
import random
import torch

import matplotlib
import matplotlib.cm


"""
Provides a set of Pytorch transforms that use OpenCV instead of PIL (Pytorch default)
for image manipulation.
"""

class Compose(object):
    # Composes transforms: transforms.Compose([transforms.RandScale([0.5, 2.0]), transforms.ToTensor()])
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, labels, intrinsics, cam_models=None, other_labels=None, transform_paras=None):
        for t in self.transforms:
            images, labels, intrinsics, cam_models, other_labels, transform_paras = t(images, labels, intrinsics, cam_models, other_labels, transform_paras)
        return images, labels, intrinsics, cam_models, other_labels, transform_paras


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __init__(self,  **kwargs):
        return
    def __call__(self, images, labels, intrinsics, cam_models=None, other_labels=None, transform_paras=None):
        if not isinstance(images, list) or not isinstance(labels, list) or not isinstance(intrinsics, list):
            raise (RuntimeError("transform.ToTensor() only handle inputs/labels/intrinsics lists."))
        if len(images) != len(intrinsics):
            raise (RuntimeError("Numbers of images and intrinsics are not matched."))
        if not isinstance(images[0], np.ndarray) or not isinstance(labels[0], np.ndarray):
            raise (RuntimeError("transform.ToTensor() only handle np.ndarray for the input and label."
                                "[eg: data readed by cv2.imread()].\n"))
        if  not isinstance(intrinsics[0], list):
            raise (RuntimeError("transform.ToTensor() only handle list for the camera intrinsics"))

        if len(images[0].shape) > 3 or len(images[0].shape) < 2:
            raise (RuntimeError("transform.ToTensor() only handle image(np.ndarray) with 3 dims or 2 dims.\n"))
        if len(labels[0].shape) > 3 or len(labels[0].shape) < 2:
            raise (RuntimeError("transform.ToTensor() only handle label(np.ndarray) with 3 dims or 2 dims.\n"))

        if len(intrinsics[0]) >4 or len(intrinsics[0]) < 3:
            raise (RuntimeError("transform.ToTensor() only handle intrinsic(list) with 3 sizes or 4 sizes.\n"))
        
        for i, img in enumerate(images):
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            images[i] = torch.from_numpy(img.transpose((2, 0, 1))).float()
        for i, lab in enumerate(labels):
            if len(lab.shape) == 2:
                lab = np.expand_dims(lab, axis=0)
            labels[i] = torch.from_numpy(lab).float()
        for i, intrinsic in enumerate(intrinsics):
            if len(intrinsic) == 3:
                intrinsic = [intrinsic[0],] + intrinsic
            intrinsics[i] = torch.tensor(intrinsic, dtype=torch.float)
        if cam_models is not None:
            for i, cam_model in enumerate(cam_models):
                cam_models[i] = torch.from_numpy(cam_model.transpose((2, 0, 1))).float() if cam_model is not None else None
        if other_labels is not None:
            for i, lab in enumerate(other_labels):
                if len(lab.shape) == 2:
                    lab = np.expand_dims(lab, axis=0)
                other_labels[i] = torch.from_numpy(lab).float()
        return images, labels, intrinsics, cam_models, other_labels, transform_paras


class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None, **kwargs):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = torch.tensor(mean).float()[:, None, None]
        self.std = torch.tensor(std).float()[:, None, None] if std is not None \
            else torch.tensor([1.0, 1.0, 1.0]).float()[:, None, None]

    def __call__(self, images, labels, intrinsics, cam_models=None, other_labels=None, transform_paras=None):
        # if self.std is None:
        #     # for t, m in zip(image, self.mean):
        #     #     t.sub(m)
        #     image = image - self.mean
        #     if ref_images is not None:
        #         for i, ref_i in enumerate(ref_images):
        #             ref_images[i] =  ref_i - self.mean
        # else:
        #     # for t, m, s in zip(image, self.mean, self.std):
        #     #     t.sub(m).div(s)
        #     image = (image - self.mean) / self.std
        #     if ref_images is not None:
        #         for i, ref_i in enumerate(ref_images):
        #             ref_images[i] =  (ref_i - self.mean) / self.std
        for i, img in enumerate(images):
            img = torch.div((img - self.mean), self.std)
            images[i] = img
        return images, labels, intrinsics, cam_models, other_labels, transform_paras


class LableScaleCanonical(object):
    """
    To solve the ambiguity observation for the mono branch, i.e. different focal length (object size) with the same depth, cameras are
    mapped to a canonical space. To mimic this, we set the focal length to a canonical one and scale the depth value. NOTE: resize the image based on the ratio can also solve
    Args:
        images: list of RGB images.
        labels: list of depth/disparity labels.
        other labels: other labels, such as instance segmentations, semantic segmentations...
    """
    def __init__(self, **kwargs):
        self.canonical_focal = kwargs['focal_length']
    
    def _get_scale_ratio(self, intrinsic):
        target_focal_x = intrinsic[0]
        label_scale_ratio = self.canonical_focal / target_focal_x
        pose_scale_ratio = 1.0
        return label_scale_ratio, pose_scale_ratio
    
    def __call__(self, images, labels, intrinsics, cam_models=None, other_labels=None, transform_paras=None):
        assert len(images[0].shape) == 3 and len(labels[0].shape) == 2
        assert labels[0].dtype == np.float32
        
        label_scale_ratio = None
        pose_scale_ratio = None

        for i in range(len(intrinsics)):
            img_i = images[i]
            label_i = labels[i] if i < len(labels) else None
            intrinsic_i = intrinsics[i].copy()
            cam_model_i = cam_models[i] if cam_models is not None and i < len(cam_models) else None

            label_scale_ratio, pose_scale_ratio = self._get_scale_ratio(intrinsic_i)

            # adjust the focal length, map the current camera to the canonical space
            intrinsics[i] = [intrinsic_i[0] * label_scale_ratio, intrinsic_i[1] * label_scale_ratio, intrinsic_i[2], intrinsic_i[3]]

            # scale the label to the canonical space
            if label_i is not None:
                labels[i] = label_i * label_scale_ratio
            
            if cam_model_i is not None:
                # As the focal length is adjusted (canonical focal length), the camera model should be re-built
                ori_h, ori_w, _ = img_i.shape
                cam_models[i] = build_camera_model(ori_h, ori_w, intrinsics[i])
            

        if transform_paras is not None:
            transform_paras.update(label_scale_factor=label_scale_ratio, focal_scale_factor=label_scale_ratio)
        
        return images, labels, intrinsics, cam_models, other_labels, transform_paras


class ResizeKeepRatio(object):
    """
    Resize and pad to a given size. Hold the aspect ratio.
    This resizing assumes that the camera model remains unchanged.
    Args:
        resize_size: predefined output size.
    """
    def __init__(self, resize_size, padding=None, ignore_label=-1, **kwargs):
        if isinstance(resize_size, int):
            self.resize_h = resize_size
            self.resize_w = resize_size
        elif isinstance(resize_size, collections.Iterable) and len(resize_size) == 2 \
                and isinstance(resize_size[0], int) and isinstance(resize_size[1], int) \
                and resize_size[0] > 0 and resize_size[1] > 0:
            self.resize_h = resize_size[0]
            self.resize_w = resize_size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))
        # self.crop_size = kwargs['crop_size']
        self.canonical_focal = kwargs['focal_length']
        
    def main_data_transform(self, image, label, intrinsic, cam_model, resize_ratio, padding, to_scale_ratio):
        """
        Resize data first and then do the padding.
        'label' will be scaled.
        """
        h, w, _ = image.shape
        reshape_h = int(resize_ratio * h)
        reshape_w = int(resize_ratio * w)

        pad_h, pad_w, pad_h_half, pad_w_half = padding
        
        # resize
        image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
        # padding
        image = cv2.copyMakeBorder(
            image, 
            pad_h_half, 
            pad_h - pad_h_half, 
            pad_w_half, 
            pad_w - pad_w_half, 
            cv2.BORDER_CONSTANT, 
            value=self.padding)

        if label is not None:
            # label = cv2.resize(label, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
            label = resize_depth_preserve(label, (reshape_h, reshape_w))
            label = cv2.copyMakeBorder(
                label, 
                pad_h_half, 
                pad_h - pad_h_half, 
                pad_w_half, 
                pad_w - pad_w_half, 
                cv2.BORDER_CONSTANT, 
                value=self.ignore_label)
            # scale the label
            label = label / to_scale_ratio
        
        # Resize, adjust principle point
        if intrinsic is not None:
            intrinsic[0] = intrinsic[0] * resize_ratio / to_scale_ratio
            intrinsic[1] = intrinsic[1] * resize_ratio / to_scale_ratio
            intrinsic[2] = intrinsic[2] * resize_ratio
            intrinsic[3] = intrinsic[3] * resize_ratio

        if cam_model is not None:
            #cam_model = cv2.resize(cam_model, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
            cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
            cam_model = cv2.copyMakeBorder(
                cam_model, 
                pad_h_half, 
                pad_h - pad_h_half, 
                pad_w_half, 
                pad_w - pad_w_half, 
                cv2.BORDER_CONSTANT, 
                value=self.ignore_label)

        # Pad, adjust the principle point
        if intrinsic is not None:
            intrinsic[2] = intrinsic[2] + pad_w_half
            intrinsic[3] = intrinsic[3] + pad_h_half
        return image, label, intrinsic, cam_model

    def get_label_scale_factor(self, image, intrinsic, resize_ratio):
        ori_h, ori_w, _ = image.shape
        # crop_h, crop_w = self.crop_size
        ori_focal = intrinsic[0]

        to_canonical_ratio = self.canonical_focal / ori_focal
        to_scale_ratio = resize_ratio / to_canonical_ratio
        return to_scale_ratio

    def __call__(self, images, labels, intrinsics, cam_models=None, other_labels=None, transform_paras=None):
        target_h, target_w, _ = images[0].shape
        resize_ratio_h = self.resize_h / target_h
        resize_ratio_w = self.resize_w / target_w
        resize_ratio = min(resize_ratio_h, resize_ratio_w)
        reshape_h = int(resize_ratio * target_h)
        reshape_w = int(resize_ratio * target_w)
        pad_h = max(self.resize_h - reshape_h, 0)
        pad_w = max(self.resize_w - reshape_w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)

        pad_info = [pad_h, pad_w, pad_h_half, pad_w_half]
        to_scale_ratio = self.get_label_scale_factor(images[0], intrinsics[0], resize_ratio)

        for i in range(len(images)):
            img = images[i]
            label = labels[i] if i < len(labels) else None
            intrinsic = intrinsics[i] if i < len(intrinsics) else None
            cam_model = cam_models[i] if cam_models is not None and i < len(cam_models) else None
            img, label, intrinsic, cam_model = self.main_data_transform(
                img, label, intrinsic, cam_model, resize_ratio, pad_info, to_scale_ratio)
            images[i] = img
            if label is not None:
                labels[i] = label
            if intrinsic is not None:
                intrinsics[i] = intrinsic
            if cam_model is not None:
                cam_models[i] = cam_model
        
        if other_labels is not None:
            
            for i, other_lab in enumerate(other_labels):
                # resize
                other_lab =  cv2.resize(other_lab, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
                # pad
                other_labels[i] =  cv2.copyMakeBorder(
                    other_lab, 
                    pad_h_half, 
                    pad_h - pad_h_half, 
                    pad_w_half, 
                    pad_w - pad_w_half, 
                    cv2.BORDER_CONSTANT, 
                    value=self.ignore_label)

        pad = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        if transform_paras is not None:
            pad_old = transform_paras['pad'] if 'pad' in transform_paras else [0,0,0,0]
            new_pad = [pad_old[0] + pad[0], pad_old[1] + pad[1], pad_old[2] + pad[2], pad_old[3] + pad[3]]
            transform_paras.update(dict(pad=new_pad))
            if 'label_scale_factor' in transform_paras:
                transform_paras['label_scale_factor'] = transform_paras['label_scale_factor'] * 1.0 / to_scale_ratio
            else:
                transform_paras.update(label_scale_factor=1.0/to_scale_ratio)
        return images, labels, intrinsics, cam_models, other_labels, transform_paras


class BGR2RGB(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __init__(self,  **kwargs):
        return
    def __call__(self, images, labels, intrinsics, cam_models=None,other_labels=None, transform_paras=None):
        for i, img in enumerate(images):
            images[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return images, labels, intrinsics, cam_models, other_labels, transform_paras
    
    
def resize_depth_preserve(depth, shape):
    """
    Resizes depth map preserving all valid depth pixels
    Multiple downsampled points can be assigned to the same pixel.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W,1]
        Resized depth map
    """
    # Store dimensions and reshapes to single column
    depth = np.squeeze(depth)
    h, w = depth.shape
    x = depth.reshape(-1)
    # Create coordinate grid
    uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
    # Filters valid points
    idx = x > 0
    crd, val = uv[idx], x[idx]
    # Downsamples coordinates
    crd[:, 0] = (crd[:, 0] * (shape[0] / h) + 0.5).astype(np.int32)
    crd[:, 1] = (crd[:, 1] * (shape[1] / w) + 0.5).astype(np.int32)
    # Filters points inside image
    idx = (crd[:, 0] < shape[0]) & (crd[:, 1] < shape[1])
    crd, val = crd[idx], val[idx]
    # Creates downsampled depth image and assigns points
    depth = np.zeros(shape)
    depth[crd[:, 0], crd[:, 1]] = val
    # Return resized depth map
    return depth


def build_camera_model(H : int, W : int, intrinsics : list) -> np.array:
    """
    Encode the camera intrinsic parameters (focal length and principle point) to a 4-channel map. 
    """
    fx, fy, u0, v0 = intrinsics
    f = (fx + fy) / 2.0
    # principle point location
    x_row = np.arange(0, W).astype(np.float32)
    x_row_center_norm = (x_row - u0) / W
    x_center = np.tile(x_row_center_norm, (H, 1)) # [H, W]

    y_col = np.arange(0, H).astype(np.float32) 
    y_col_center_norm = (y_col - v0) / H
    y_center = np.tile(y_col_center_norm, (W, 1)).T

    # FoV
    fov_x = np.arctan(x_center / (f / W))
    fov_y =  np.arctan(y_center/ (f / H))

    cam_model = np.stack([x_center, y_center, fov_x, fov_y], axis=2)
    return cam_model

def gray_to_colormap(img, cmap='rainbow'):
    """
    Transfer gray map to matplotlib colormap
    """
    assert img.ndim == 2

    img[img<0] = 0
    mask_invalid = img < 1e-10
    img = img / (img.max() + 1e-8)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0
    return colormap