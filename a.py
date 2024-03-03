import cv2 as cv
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import cpu_count
from typing import Tuple, List, Union
import torch

device = "cuda"

def calculate_gradients(
    normals: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    horizontal_angle_map = torch.arccos(torch.clip(normals[:, :, 0], -1, 1))
    left_gradients = torch.zeros(normals.shape[:2]).to(device)
    left_gradients[mask != 0] = (1 - torch.sin(horizontal_angle_map[mask != 0])) * torch.sign(
        horizontal_angle_map[mask != 0] - torch.pi / 2
    )

    vertical_angle_map = torch.arccos(torch.clip(normals[:, :, 1], -1, 1))
    top_gradients = torch.zeros(normals.shape[:2]).to(device)
    top_gradients[mask != 0] = -(1 - torch.sin(vertical_angle_map[mask != 0])) * torch.sign(
        vertical_angle_map[mask != 0] - torch.pi / 2
    )

    return left_gradients, top_gradients


def integrate_gradient_field(
    gradient_field: torch.Tensor, axis: int, mask: torch.Tensor
) -> torch.Tensor:
    heights = torch.zeros(gradient_field.shape).to(device)

    for d1 in range(heights.shape[1 - axis]):
        sum_value = 0
        for d2 in range(heights.shape[axis]):
            coordinates = (d1, d2) if axis == 1 else (d2, d1)

            if mask[coordinates] != 0:
                sum_value = sum_value + gradient_field[coordinates]
                heights[coordinates] = sum_value
            else:
                sum_value = 0

    return heights


def calculate_heights(
    left_gradients: torch.Tensor, top_gradients, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    left_heights = integrate_gradient_field(left_gradients, 1, mask)
    right_heights = torch.fliplr(
        integrate_gradient_field(torch.fliplr(-left_gradients), 1, torch.fliplr(mask))
    )
    top_heights = integrate_gradient_field(top_gradients, 0, mask)
    bottom_heights = torch.flipud(
        integrate_gradient_field(torch.flipud(-top_gradients), 0, torch.flipud(mask))
    )
    return left_heights, right_heights, top_heights, bottom_heights


def combine_heights(*heights: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.stack(heights, axis=0), axis=0)


def rotate(matrix: torch.Tensor, angle: float) -> torch.Tensor:
    h, w = matrix.cpu().detach().numpy().shape[:2]
    center = (w / 2, h / 2)

    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    corners = cv.transform(
        torch.array([[[0, 0], [w, 0], [w, h], [0, h]]]), rotation_matrix
    )[0]

    _, _, w, h = cv.boundingRect(corners)

    rotation_matrix[0, 2] += w / 2 - center[0]
    rotation_matrix[1, 2] += h / 2 - center[1]
    result = cv.warpAffine(matrix, rotation_matrix, (w, h), flags=cv.INTER_LINEAR)

    return torch.from_numpy(result).to(device)


def rotate_vector_field_normals(normals: torch.Tensor, angle: float) -> torch.Tensor:
    angle = torch.radians(angle)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    rotated_normals = torch.empty_like(normals)
    rotated_normals[:, :, 0] = (
        normals[:, :, 0] * cos_angle - normals[:, :, 1] * sin_angle
    )
    rotated_normals[:, :, 1] = (
        normals[:, :, 0] * sin_angle + normals[:, :, 1] * cos_angle
    )

    return rotated_normals


def centered_crop(image: torch.Tensor, target_resolution: Tuple[int, int]) -> torch.Tensor:
    return image[
        (image.shape[0] - target_resolution[0])
        // 2 : (image.shape[0] - target_resolution[0])
        // 2
        + target_resolution[0],
        (image.shape[1] - target_resolution[1])
        // 2 : (image.shape[1] - target_resolution[1])
        // 2
        + target_resolution[1],
    ]


def integrate_vector_field(
    vector_field: torch.Tensor,
    mask: torch.Tensor,
    target_iteration_count: int,
    thread_count: int,
) -> torch.Tensor:
    shape = vector_field.shape[:2]
    angles = torch.linspace(0, 90, target_iteration_count, endpoint=False)

    def integrate_vector_field_angles(angles: List[float]) -> torch.Tensor:
        all_combined_heights = torch.zeros(shape).to(device)

        for angle in angles:
            rotated_vector_field = rotate_vector_field_normals(
                rotate(vector_field, angle), angle
            )
            rotated_mask = rotate(mask, angle)

            left_gradients, top_gradients = calculate_gradients(
                rotated_vector_field, rotated_mask
            )
            (
                left_heights,
                right_heights,
                top_heights,
                bottom_heights,
            ) = calculate_heights(left_gradients, top_gradients, rotated_mask)

            combined_heights = combine_heights(
                left_heights, right_heights, top_heights, bottom_heights
            )
            combined_heights = centered_crop(rotate(combined_heights, -angle), shape)
            all_combined_heights += combined_heights / len(angles)

        return all_combined_heights

    isotropic_height = integrate_vector_field_angles(angles)

    return isotropic_height


def estimate_height_map(
    normal_map: torch.Tensor,
    mask: Union[torch.Tensor, None] = None,
    height_divisor: float = 1,
    target_iteration_count: int = 250,
    thread_count: int = cpu_count(),
    raw_values: bool = False,
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones(normal_map.shape[:2], dtype=torch.uint8).to(device)

    normals = ((normal_map[:, :, :3].to(torch.float64).to(device) / 255) - 0.5) * 2
    heights = integrate_vector_field(
        normals, mask, target_iteration_count, thread_count
    )

    if raw_values:
        return heights

    heights /= height_divisor
    heights[mask > 0] += 1 / 2
    heights[mask == 0] = 1 / 2

    heights *= 2**16 - 1

    if torch.min(heights) < 0 or torch.max(heights) > 2**16 - 1:
        raise OverflowError("Height values are clipping.")

    heights = torch.clip(heights, 0, 2**16 - 1)
    heights = heights.to(torch.uint16)

    return heights