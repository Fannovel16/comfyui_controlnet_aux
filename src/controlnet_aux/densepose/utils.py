from typing import Optional, Tuple
import math
import numpy as np
from enum import IntEnum, unique
from typing import List, Tuple, Union
import torch
from torch import device
from torch.nn import functional as F
import logging
import einops
import cv2
import custom_detectron2.data.transforms as T

Image = np.ndarray
Boxes = torch.Tensor
N_PART_LABELS = 24
ImageSizeType = Tuple[int, int]
_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]
IntTupleBox = Tuple[int, int, int, int]

class BoxMode(IntEnum):
    """
    Enum of different ways to represent a box.
    """

    XYXY_ABS = 0
    """
    (x0, y0, x1, y1) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    """
    XYWH_ABS = 1
    """
    (x0, y0, w, h) in absolute floating points coordinates.
    """
    XYXY_REL = 2
    """
    Not yet supported!
    (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
    """
    XYWH_REL = 3
    """
    Not yet supported!
    (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
    """
    XYWHA_ABS = 4
    """
    (xc, yc, w, h, a) in absolute floating points coordinates.
    (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
    """

    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) == 4 or len(box) == 5, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 4 or 5"
            )
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            else:
                arr = box.clone()

        assert to_mode not in [BoxMode.XYXY_REL, BoxMode.XYWH_REL] and from_mode not in [
            BoxMode.XYXY_REL,
            BoxMode.XYWH_REL,
        ], "Relative mode not yet supported!"

        if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
            assert (
                arr.shape[-1] == 5
            ), "The last dimension of input shape must be 5 for XYWHA format"
            original_dtype = arr.dtype
            arr = arr.double()

            w = arr[:, 2]
            h = arr[:, 3]
            a = arr[:, 4]
            c = torch.abs(torch.cos(a * math.pi / 180.0))
            s = torch.abs(torch.sin(a * math.pi / 180.0))
            # This basically computes the horizontal bounding rectangle of the rotated box
            new_w = c * w + s * h
            new_h = c * h + s * w

            # convert center to top-left corner
            arr[:, 0] -= new_w / 2.0
            arr[:, 1] -= new_h / 2.0
            # bottom-right corner
            arr[:, 2] = arr[:, 0] + new_w
            arr[:, 3] = arr[:, 1] + new_h

            arr = arr[:, :4].to(dtype=original_dtype)
        elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
            original_dtype = arr.dtype
            arr = arr.double()
            arr[:, 0] += arr[:, 2] / 2.0
            arr[:, 1] += arr[:, 3] / 2.0
            angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
            arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
        else:
            if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
                arr[:, 2] += arr[:, 0]
                arr[:, 3] += arr[:, 1]
            elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
                arr[:, 2] -= arr[:, 0]
                arr[:, 3] -= arr[:, 1]
            else:
                raise NotImplementedError(
                    "Conversion from BoxMode {} to {} is not supported yet".format(
                        from_mode, to_mode
                    )
                )

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr

class MatrixVisualizer:
    """
    Base visualizer for matrix data
    """

    def __init__(
        self,
        inplace=True,
        cmap=cv2.COLORMAP_PARULA,
        val_scale=1.0,
        alpha=0.7,
        interp_method_matrix=cv2.INTER_LINEAR,
        interp_method_mask=cv2.INTER_NEAREST,
    ):
        self.inplace = inplace
        self.cmap = cmap
        self.val_scale = val_scale
        self.alpha = alpha
        self.interp_method_matrix = interp_method_matrix
        self.interp_method_mask = interp_method_mask

    def visualize(self, image_bgr, mask, matrix, bbox_xywh):
        self._check_image(image_bgr)
        self._check_mask_matrix(mask, matrix)
        if self.inplace:
            image_target_bgr = image_bgr
        else:
            image_target_bgr = image_bgr * 0
        x, y, w, h = [int(v) for v in bbox_xywh]
        if w <= 0 or h <= 0:
            return image_bgr  
        mask, matrix = self._resize(mask, matrix, w, h)
        mask_bg = np.tile((mask == 0)[:, :, np.newaxis], [1, 1, 3])
        matrix_scaled = matrix.astype(np.float32) * self.val_scale
        _EPSILON = 1e-6
        if np.any(matrix_scaled > 255 + _EPSILON):
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Matrix has values > {255 + _EPSILON} after " f"scaling, clipping to [0..255]"
            )
        matrix_scaled_8u = matrix_scaled.clip(0, 255).astype(np.uint8)
        matrix_vis = cv2.applyColorMap(matrix_scaled_8u, self.cmap)
        #print(mask_bg.shape, matrix_vis.shape, image_target_bgr.shape, image_target_bgr[y : y + h, x : x + w, :].shape)
        matrix_vis[mask_bg] = image_target_bgr[y : y + h, x : x + w, :][mask_bg]
        image_target_bgr[y : y + h, x : x + w, :] = (
            image_target_bgr[y : y + h, x : x + w, :] * (1.0 - self.alpha) + matrix_vis * self.alpha
        )
        return image_target_bgr.astype(np.uint8)

    def _resize(self, mask, matrix, w, h):
        if (w != mask.shape[1]) or (h != mask.shape[0]):
            mask = cv2.resize(mask, (w, h), self.interp_method_mask)
        if (w != matrix.shape[1]) or (h != matrix.shape[0]):
            matrix = cv2.resize(matrix, (w, h), self.interp_method_matrix)
        return mask, matrix

    def _check_image(self, image_rgb):
        assert len(image_rgb.shape) == 3
        assert image_rgb.shape[2] == 3
        assert image_rgb.dtype == np.uint8

    def _check_mask_matrix(self, mask, matrix):
        assert len(matrix.shape) == 2
        assert len(mask.shape) == 2
        assert mask.dtype == np.uint8

class DensePoseResultsVisualizer:
    def visualize(
        self,
        image_bgr: Image,
        results,
    ) -> Image:
        context = self.create_visualization_context(image_bgr)
        for i, result in enumerate(results):
            boxes_xywh, labels, uv = result
            iuv_array = torch.cat(
                (labels[None].type(torch.float32), uv * 255.0)
            ).type(torch.uint8)
            self.visualize_iuv_arr(context, iuv_array.cpu().numpy(), boxes_xywh)
        image_bgr = self.context_to_image_bgr(context)
        return image_bgr

    def create_visualization_context(self, image_bgr: Image):
        return image_bgr

    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh) -> None:
        pass

    def context_to_image_bgr(self, context):
        return context

    def get_image_bgr_from_context(self, context):
        return context

class DensePoseMaskedColormapResultsVisualizer(DensePoseResultsVisualizer):
    def __init__(
        self,
        data_extractor,
        segm_extractor,
        inplace=True,
        cmap=cv2.COLORMAP_PARULA,
        alpha=0.7,
        val_scale=1.0,
        **kwargs,
    ):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=val_scale, alpha=alpha
        )
        self.data_extractor = data_extractor
        self.segm_extractor = segm_extractor

    def context_to_image_bgr(self, context):
        return context

    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh) -> None:
        image_bgr = self.get_image_bgr_from_context(context)
        matrix = self.data_extractor(iuv_arr)
        segm = self.segm_extractor(iuv_arr)
        mask = np.zeros(matrix.shape, dtype=np.uint8)
        mask[segm > 0] = 1
        image_bgr = self.mask_visualizer.visualize(image_bgr, mask, matrix, bbox_xywh)


def _extract_i_from_iuvarr(iuv_arr):
    return iuv_arr[0, :, :]


def _extract_u_from_iuvarr(iuv_arr):
    return iuv_arr[1, :, :]


def _extract_v_from_iuvarr(iuv_arr):
    return iuv_arr[2, :, :]

def make_int_box(box: torch.Tensor) -> IntTupleBox:
    int_box = [0, 0, 0, 0]
    int_box[0], int_box[1], int_box[2], int_box[3] = tuple(box.long().tolist())
    return int_box[0], int_box[1], int_box[2], int_box[3]

def densepose_chart_predictor_output_to_result_with_confidences(
    boxes: Boxes,
    coarse_segm,
    fine_segm,
    u, v

):
    boxes_xyxy_abs = boxes.clone()
    boxes_xywh_abs = BoxMode.convert(boxes_xyxy_abs, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    box_xywh = make_int_box(boxes_xywh_abs[0])

    labels = resample_fine_and_coarse_segm_tensors_to_bbox(fine_segm, coarse_segm, box_xywh).squeeze(0)
    uv = resample_uv_tensors_to_bbox(u, v, labels, box_xywh)
    confidences = []
    return box_xywh, labels, uv

def resample_fine_and_coarse_segm_tensors_to_bbox(
    fine_segm: torch.Tensor, coarse_segm: torch.Tensor, box_xywh_abs: IntTupleBox
):
    """
    Resample fine and coarse segmentation tensors to the given
    bounding box and derive labels for each pixel of the bounding box

    Args:
        fine_segm: float tensor of shape [1, C, Hout, Wout]
        coarse_segm: float tensor of shape [1, K, Hout, Wout]
        box_xywh_abs (tuple of 4 int): bounding box given by its upper-left
            corner coordinates, width (W) and height (H)
    Return:
        Labels for each pixel of the bounding box, a long tensor of size [1, H, W]
    """
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    # coarse segmentation
    coarse_segm_bbox = F.interpolate(
        coarse_segm,
        (h, w),
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)
    # combined coarse and fine segmentation
    labels = (
        F.interpolate(fine_segm, (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
        * (coarse_segm_bbox > 0).long()
    )
    return labels

def resample_uv_tensors_to_bbox(
    u: torch.Tensor,
    v: torch.Tensor,
    labels: torch.Tensor,
    box_xywh_abs: IntTupleBox,
) -> torch.Tensor:
    """
    Resamples U and V coordinate estimates for the given bounding box

    Args:
        u (tensor [1, C, H, W] of float): U coordinates
        v (tensor [1, C, H, W] of float): V coordinates
        labels (tensor [H, W] of long): labels obtained by resampling segmentation
            outputs for the given bounding box
        box_xywh_abs (tuple of 4 int): bounding box that corresponds to predictor outputs
    Return:
       Resampled U and V coordinates - a tensor [2, H, W] of float
    """
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    u_bbox = F.interpolate(u, (h, w), mode="bilinear", align_corners=False)
    v_bbox = F.interpolate(v, (h, w), mode="bilinear", align_corners=False)
    uv = torch.zeros([2, h, w], dtype=torch.float32, device=u.device)
    for part_id in range(1, u_bbox.size(1)):
        uv[0][labels == part_id] = u_bbox[0, part_id][labels == part_id]
        uv[1][labels == part_id] = v_bbox[0, part_id][labels == part_id]
    return uv

def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

INPUT_SIZE = (1024, 1024)
def infernce(model, np_image, device="cuda"):
    civitai_vis = DensePoseMaskedColormapResultsVisualizer(cmap=cv2.COLORMAP_PARULA, alpha=1, data_extractor=_extract_i_from_iuvarr, segm_extractor=_extract_i_from_iuvarr, val_scale = 255.0 / N_PART_LABELS)
    magic_animate_vis = DensePoseMaskedColormapResultsVisualizer(cmap=cv2.COLORMAP_PARULA, alpha=1, data_extractor=_extract_i_from_iuvarr, segm_extractor=_extract_i_from_iuvarr, val_scale = 255.0 / N_PART_LABELS)

    extractor = densepose_chart_predictor_output_to_result_with_confidences #aka ToChartResultConverterWithConfidences
    #resized_image = AUG.get_transform(np_image).apply_image(np_image).copy()
    resized_image, r = preprocess(np_image, INPUT_SIZE) #CHW
    height, width = resized_image.shape[1:]

    pred_xyxy_boxes, corase_segm, fine_segm, u, v = model(torch.from_numpy(resized_image).to(device))

    densepose_results = []
    for i in range(len(pred_xyxy_boxes)):
        #For some reasons, the boxes can be out-of-bound
        if pred_xyxy_boxes[i,0] + pred_xyxy_boxes[i,2] > width: 
            pred_xyxy_boxes[i,2] = width
        if pred_xyxy_boxes[i,1] + pred_xyxy_boxes[i,2] > height: 
            pred_xyxy_boxes[i,3] = height
        box_xywh, labels, uv = extractor(pred_xyxy_boxes[i:i+1], corase_segm[i:i+1], fine_segm[i:i+1], u[i:i+1], v[i:i+1])
        #print("image, box, labels, uv: ", image.shape, box_xywh, labels.shape, uv.shape)
        densepose_results.append((box_xywh, labels, uv))

    hint_image_canvas = np.zeros([height, width], dtype=np.uint8)
    hint_image_canvas = np.tile(hint_image_canvas[:, :, np.newaxis], [1, 1, 3])
    civitai_hint_image = civitai_vis.visualize(hint_image_canvas, densepose_results)
    civitai_hint_image = cv2.cvtColor(civitai_hint_image, cv2.COLOR_BGR2RGB)

    magic_animate_hint_image = magic_animate_vis.visualize(hint_image_canvas, densepose_results)
    magic_animate_hint_image = cv2.cvtColor(magic_animate_hint_image, cv2.COLOR_BGR2RGB)
    magic_animate_hint_image[:, :, 0][magic_animate_hint_image[:, :, 0] == 0] = 68
    magic_animate_hint_image[:, :, 1][magic_animate_hint_image[:, :, 1] == 0] = 1
    magic_animate_hint_image[:, :, 2][magic_animate_hint_image[:, :, 2] == 0] = 84
    
    main_h, main_w = int(np_image.shape[0] * r), int(np_image.shape[1] * r)
    civitai_hint_image = civitai_hint_image[:main_h, :main_w]
    civitai_hint_image = cv2.resize(civitai_hint_image, (0, 0), fx=1/r, fy=1/r, interpolation=cv2.INTER_LINEAR)
    magic_animate_hint_image = magic_animate_hint_image[:main_h, :main_w]
    magic_animate_hint_image = cv2.resize(magic_animate_hint_image, (0, 0), fx=1/r, fy=1/r, interpolation=cv2.INTER_LINEAR)
    return civitai_hint_image, magic_animate_hint_image