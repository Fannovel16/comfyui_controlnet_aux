import torch
import torchvision.transforms as T
import torch.utils.data as data
import torch.nn as nn
from pathlib import Path
from functools import partial
from custom_controlnet_aux.diffusion_edge.denoising_diffusion_pytorch.utils import exists, convert_image_to_fn, normalize_to_neg_one_to_one
from PIL import Image, ImageDraw
import torch.nn.functional as F
import math
import torchvision.transforms.functional as F2
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from typing import Any, Callable, Optional, Tuple
import os
import pickle
import numpy as np
import copy
import albumentations
from torchvision.transforms.functional import InterpolationMode

def get_imgs_list(imgs_dir):
    imgs_list = os.listdir(imgs_dir)
    imgs_list.sort()
    return [os.path.join(imgs_dir, f) for f in imgs_list if f.endswith('.jpg') or f.endswith('.JPG')or f.endswith('.png') or f.endswith('.pgm') or f.endswith('.ppm')]


def fit_img_postfix(img_path):
    if not os.path.exists(img_path) and img_path.endswith(".jpg"):
        img_path = img_path[:-4] + ".png"
    if not os.path.exists(img_path) and img_path.endswith(".png"):
        img_path = img_path[:-4] + ".jpg"
    return img_path


class AdaptEdgeDataset(data.Dataset):
    def __init__(
        self,
        data_root,
        # mask_folder,
        image_size,
        exts = ['png', 'jpg'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        normalize_to_neg_one_to_one=True,
        split='train',
        # inter_type='bicubic',
        # down=4,
        threshold=0.3, use_uncertainty=False
    ):
        super().__init__()
        # self.img_folder = Path(img_folder)
        # self.edge_folder = Path(os.path.join(data_root, f'gt_imgs'))
        # self.img_folder = Path(os.path.join(data_root, f'imgs'))
        # self.edge_folder = Path(os.path.join(data_root, "edge", "aug"))
        # self.img_folder = Path(os.path.join(data_root, "image", "aug"))
        self.data_root = data_root
        self.image_size = image_size

        # self.edge_paths = [p for ext in exts for p in self.edge_folder.rglob(f'*.{ext}')]
        # self.img_paths = [(self.img_folder / item.parent.name / f'{item.stem}.jpg') for item in self.edge_paths]
        # self.img_paths = [(self.img_folder / f'{item.stem}.jpg') for item in self.edge_paths]

        self.threshold = threshold * 256
        self.use_uncertainty = use_uncertainty
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else Identity()

        # self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        # self.random_crop = RandomCrop(size=image_size)
        # self.transform = Compose([
        #     # Lambda(maybe_convert_fn),
        #     # Resize(image_size, interpolation=3, interpolation2=0),
        #     Resize(image_size, interpolation=InterpolationMode.BILINEAR, interpolation2=InterpolationMode.NEAREST),
        #     RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
        #     # RandomCrop(image_size),
        #     ToTensor()
        # ])
        self.data_list = self.build_list()

        self.transform = transforms.Compose([
            # Resize(self.image_size, interpolation=InterpolationMode.BILINEAR, interpolation2=InterpolationMode.NEAREST),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.data_list)


    def read_img(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        raw_width, raw_height = img.size
        # width = int(raw_width / 32) * 32
        # height = int(raw_height / 32) * 32
        # img = img.resize((width, height), Image.Resampling.BILINEAR)
        # # print("img.size:", img.size)
        # img = self.transform(img)

        return img, (raw_width, raw_height)

    def read_lb(self, lb_path):
        lb_data = Image.open(lb_path)

        width, height = lb_data.size
        width = int(width / 32) * 32
        height = int(height / 32) * 32
        lb_data = lb_data.resize((width, height), Image.Resampling.BILINEAR)
        # print("lb_data.size:", lb_data.size)
        lb = np.array(lb_data, dtype=np.float32)
        if lb.ndim == 3:
            lb = np.squeeze(lb[:, :, 0])
        assert lb.ndim == 2
        threshold = self.threshold
        lb = lb[np.newaxis, :, :]

        lb[lb == 0] = 0

        # ---------- important ----------
        if self.use_uncertainty:
            lb[np.logical_and(lb > 0, lb < threshold)] = 2
        else:
            lb[np.logical_and(lb > 0, lb < threshold)] /= 255.

        lb[lb >= threshold] = 1
        return lb

    def build_list(self):
        data_root = os.path.abspath(self.data_root)
        images_path = os.path.join(data_root, 'image', "raw")
        labels_path = os.path.join(data_root, 'edge', "raw")

        samples = []
        for directory_name in os.listdir(images_path):
            image_directories = os.path.join(images_path, directory_name)
            for file_name_ext in os.listdir(image_directories):
                file_name = os.path.basename(file_name_ext)
                image_path = fit_img_postfix(os.path.join(images_path, directory_name, file_name))
                lb_path = fit_img_postfix(os.path.join(labels_path, directory_name, file_name))
                samples.append((image_path, lb_path))
        return samples

    def __getitem__(self, index):
        img_path, edge_path = self.data_list[index]
        # edge_path = self.edge_paths[index]
        # img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)

        img, raw_size = self.read_img(img_path)
        edge = self.read_lb(edge_path)

        # print("-------hhhhhhhhhhhhh--------:", img.shape, edge.shape)
        # edge = Image.open(edge_path).convert('L')
        # # default to score-sde preprocessing
        # mask = Image.open(img_path).convert('RGB')
        # edge, img = self.transform(edge, mask)
        if self.normalize_to_neg_one_to_one:   # transform to [-1, 1]
            edge = normalize_to_neg_one_to_one(edge)
            img = normalize_to_neg_one_to_one(img)
        return {'image': edge, 'cond': img, 'raw_size': raw_size, 'img_name': img_name}

class EdgeDataset(data.Dataset):
    def __init__(
        self,
        data_root,
        # mask_folder,
        image_size,
        exts = ['png', 'jpg'],
        augment_horizontal_flip = True,
        convert_image_to = None,
        normalize_to_neg_one_to_one=True,
        split='train',
        # inter_type='bicubic',
        # down=4,
        threshold=0.3, use_uncertainty=False, cfg={}
    ):
        super().__init__()
        # self.img_folder = Path(img_folder)
        # self.edge_folder = Path(os.path.join(data_root, f'gt_imgs'))
        # self.img_folder = Path(os.path.join(data_root, f'imgs'))
        # self.edge_folder = Path(os.path.join(data_root, "edge", "aug"))
        # self.img_folder = Path(os.path.join(data_root, "image", "aug"))
        self.data_root = data_root
        self.image_size = image_size

        # self.edge_paths = [p for ext in exts for p in self.edge_folder.rglob(f'*.{ext}')]
        # self.img_paths = [(self.img_folder / item.parent.name / f'{item.stem}.jpg') for item in self.edge_paths]
        # self.img_paths = [(self.img_folder / f'{item.stem}.jpg') for item in self.edge_paths]

        self.threshold = threshold * 255
        self.use_uncertainty = use_uncertainty
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else Identity()

        self.data_list = self.build_list()

        # self.transform = Compose([
        #     Resize(image_size),
        #     RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
        #     ToTensor()
        # ])
        crop_type = cfg.get('crop_type') if 'crop_type' in cfg else 'rand_crop'
        if crop_type == 'rand_crop':
            self.transform = Compose([
                RandomCrop(image_size),
                RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
                ToTensor()
            ])
        elif crop_type == 'rand_resize_crop':
            self.transform = Compose([
                RandomResizeCrop(image_size),
                RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
                ToTensor()
            ])
        print("crop_type:", crop_type)

    def __len__(self):
        return len(self.data_list)


    def read_img(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        raw_width, raw_height = img.size
        # width = int(raw_width / 32) * 32
        # height = int(raw_height / 32) * 32
        # img = img.resize((width, height), Image.Resampling.BILINEAR)
        # # print("img.size:", img.size)
        # img = self.transform(img)

        return img, (raw_width, raw_height)

    def read_lb(self, lb_path):
        lb_data = Image.open(lb_path).convert('L')
        lb = np.array(lb_data).astype(np.float32)
        # width, height = lb_data.size
        # width = int(width / 32) * 32
        # height = int(height / 32) * 32
        # lb_data = lb_data.resize((width, height), Image.Resampling.BILINEAR)
        # print("lb_data.size:", lb_data.size)
        # lb = np.array(lb_data, dtype=np.float32)
        # if lb.ndim == 3:
        #     lb = np.squeeze(lb[:, :, 0])
        # assert lb.ndim == 2
        threshold = self.threshold
        # lb = lb[np.newaxis, :, :]
        # lb[lb == 0] = 0

        # ---------- important ----------
        # if self.use_uncertainty:
        #     lb[np.logical_and(lb > 0, lb < threshold)] = 2
        # else:
        #     lb[np.logical_and(lb > 0, lb < threshold)] /= 255.

        lb[lb >= threshold] = 255
        lb = Image.fromarray(lb.astype(np.uint8))
        return lb

    def build_list(self):
        data_root = os.path.abspath(self.data_root)
        images_path = os.path.join(data_root, 'image')
        labels_path = os.path.join(data_root, 'edge')

        samples = []
        for directory_name in os.listdir(images_path):
            image_directories = os.path.join(images_path, directory_name)
            for file_name_ext in os.listdir(image_directories):
                file_name = os.path.basename(file_name_ext)
                image_path = fit_img_postfix(os.path.join(images_path, directory_name, file_name))
                lb_path = fit_img_postfix(os.path.join(labels_path, directory_name, file_name))
                samples.append((image_path, lb_path))
        return samples

    def __getitem__(self, index):
        img_path, edge_path = self.data_list[index]
        # edge_path = self.edge_paths[index]
        # img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)

        img, raw_size = self.read_img(img_path)
        edge = self.read_lb(edge_path)
        img, edge = self.transform(img, edge)

        # print("-------hhhhhhhhhhhhh--------:", img.shape, edge.shape)
        # edge = Image.open(edge_path).convert('L')
        # # default to score-sde preprocessing
        # mask = Image.open(img_path).convert('RGB')
        # edge, img = self.transform(edge, mask)
        if self.normalize_to_neg_one_to_one:   # transform to [-1, 1]
            edge = normalize_to_neg_one_to_one(edge)
            img = normalize_to_neg_one_to_one(img)
        return {'image': edge, 'cond': img, 'raw_size': raw_size, 'img_name': img_name}

class EdgeDatasetTest(data.Dataset):
    def __init__(
        self,
        data_root,
        # mask_folder,
        image_size,
        exts = ['png', 'jpg'],
        convert_image_to = None,
        normalize_to_neg_one_to_one=True,
    ):
        super().__init__()

        self.data_root = data_root
        self.image_size = image_size
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else Identity()

        self.data_list = self.build_list()

        self.transform = Compose([
            ToTensor()
        ])

    def __len__(self):
        return len(self.data_list)


    def read_img(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        raw_width, raw_height = img.size


        return img, (raw_width, raw_height)

    def read_lb(self, lb_path):
        lb_data = Image.open(lb_path).convert('L')
        lb = np.array(lb_data).astype(np.float32)

        threshold = self.threshold


        lb[lb >= threshold] = 255
        lb = Image.fromarray(lb.astype(np.uint8))
        return lb

    def build_list(self):
        data_root = os.path.abspath(self.data_root)
        # images_path = os.path.join(data_root)
        images_path = data_root
        samples = get_imgs_list(images_path)
        return samples

    def __getitem__(self, index):
        img_path = self.data_list[index]
        # edge_path = self.edge_paths[index]
        # img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)

        img, raw_size = self.read_img(img_path)

        img = self.transform(img)
        if self.normalize_to_neg_one_to_one:   # transform to [-1, 1]
            img = normalize_to_neg_one_to_one(img)
        return {'cond': img, 'raw_size': raw_size, 'img_name': img_name}


class Identity(nn.Identity):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__(*args, **kwargs)

    def forward(self, input, target):
        return input, target

class Resize(T.Resize):
    def __init__(self, size, interpolation2=None, **kwargs):
        super().__init__(size, **kwargs)
        if interpolation2 is None:
            self.interpolation2 = self.interpolation
        else:
            self.interpolation2 = interpolation2

    def forward(self, img, target=None):
        if target is None:
            img = F2.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
            return img
        else:
            img = F2.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
            target = F2.resize(target, self.size, self.interpolation2, self.max_size, self.antialias)
            return img, target

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def forward(self, img, target=None):
        if target is None:
            if torch.rand(1) < self.p:
                img = F2.hflip(img)
            return img
        else:
            if torch.rand(1) < self.p:
                img = F2.hflip(img)
                target = F2.hflip(target)
            return img, target

class CenterCrop(T.CenterCrop):
    def __init__(self, size):
        super().__init__(size)

    def forward(self, img, target=None):
        if target is None:
            img = F2.center_crop(img, self.size)
            return img
        else:
            img = F2.center_crop(img, self.size)
            target = F2.center_crop(target, self.size)
            return img, target

class RandomCrop(T.RandomCrop):
    def __init__(self, size, **kwargs):
        super().__init__(size, **kwargs)

    def single_forward(self, img, i, j, h, w):
        if self.padding is not None:
            img = F2.pad(img, self.padding, self.fill, self.padding_mode)
        width, height = F2.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F2.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F2.pad(img, padding, self.fill, self.padding_mode)

        return F2.crop(img, i, j, h, w)

    def forward(self, img, target=None):
        i, j, h, w = self.get_params(img, self.size)
        if target is None:
            img = self.single_forward(img, i, j, h, w)
            return img
        else:
            img = self.single_forward(img, i, j, h, w)
            target = self.single_forward(target, i, j, h, w)
            return img, target

class RandomResizeCrop(T.RandomResizedCrop):
    def __init__(self, size, scale=(0.25, 1.0), **kwargs):
        super().__init__(size, scale, **kwargs)

    # def single_forward(self, img, i, j, h, w):
    #     if self.padding is not None:
    #         img = F2.pad(img, self.padding, self.fill, self.padding_mode)
    #     width, height = F2.get_image_size(img)
    #     # pad the width if needed
    #     if self.pad_if_needed and width < self.size[1]:
    #         padding = [self.size[1] - width, 0]
    #         img = F2.pad(img, padding, self.fill, self.padding_mode)
    #     # pad the height if needed
    #     if self.pad_if_needed and height < self.size[0]:
    #         padding = [0, self.size[0] - height]
    #         img = F2.pad(img, padding, self.fill, self.padding_mode)
    #
    #     return F2.crop(img, i, j, h, w)

    def single_forward(self, img, i, j, h, w, interpolation=InterpolationMode.BILINEAR):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        # i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F2.resized_crop(img, i, j, h, w, self.size, interpolation)

    def forward(self, img, target=None):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if target is None:
            img = self.single_forward(img, i, j, h, w)
            return img
        else:
            img = self.single_forward(img, i, j, h, w)
            target = self.single_forward(target, i, j, h, w, interpolation=InterpolationMode.NEAREST)
            return img, target

class ToTensor(T.ToTensor):
    def __init__(self):
        super().__init__()

    def __call__(self, img, target=None):
        if target is None:
            img = F2.to_tensor(img)
            return img
        else:
            img = F2.to_tensor(img)
            target = F2.to_tensor(target)
            return img, target

class Lambda(T.Lambda):
    """Apply a user-defined lambda as a transform. This transform does not support torchscript.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        super().__init__(lambd)

    def __call__(self, img, target=None):
        if target is None:
            return self.lambd(img)
        else:
            return self.lambd(img), self.lambd(target)

class Compose(T.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, img, target=None):
        if target is None:
            for t in self.transforms:
                img = t(img)
            return img
        else:
            for t in self.transforms:
                img, target = t(img, target)
            return img, target


if __name__ == '__main__':
    dataset = CIFAR10(
        img_folder='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/cifar-10-python',
        augment_horizontal_flip=False
    )
    # dataset = CityscapesDataset(
    #     # img_folder='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/CelebAHQ/celeba_hq_256',
    #     data_root='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/Cityscapes/',
    #     # data_root='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/ADEChallengeData2016/',
    #     image_size=[512, 1024],
    #     exts = ['png'],
    #     augment_horizontal_flip = False,
    #     convert_image_to = None,
    #     normalize_to_neg_one_to_one=True,
    #     )
    # dataset = SRDataset(
    #     img_folder='/media/huang/ZX3 512G/data/DIV2K/DIV2K_train_HR',
    #     image_size=[512, 512],
    # )
    # dataset = InpaintDataset(
    #     img_folder='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/CelebAHQ/celeba_hq_256',
    #     image_size=[256, 256],
    #     augment_horizontal_flip = True
    # )
    dataset = EdgeDataset(
        data_root='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/BSDS',
        image_size=[320, 320],
    )
    for i in range(len(dataset)):
        d = dataset[i]
        mask = d['cond']
        print(mask.max())
    dl = data.DataLoader(dataset, batch_size=2, shuffle=False, pin_memory=True, num_workers=0)


    dataset_builder = tfds.builder('cifar10')
    split = 'train'
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    dataset_builder.download_and_prepare()
    ds = dataset_builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config)
    pause = 0