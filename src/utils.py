import math
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .constants import FIELD_HEIGHT, FIELD_WIDTH


def getFieldPoints(scaled_width: int, scaled_height: int) -> np.ndarray:
    """Generate 91 scaled field points for a basketball court.

    Args:
        scaled_width (int): Width of the basketball court to be scaled onto.
        scaled_height (int): Height of the basketball court to be scaled onto.

    Returns:
        np.ndarray: Scaled field points for a basketball court.
    """
    points = []
    u0 = 175
    r = 30
    u = u0
    s = 0
    for _ in range(0, 7):
        for i in range(0, 13):
            points.append([i * FIELD_WIDTH / 12, FIELD_HEIGHT - s])
        s += u
        u += r

    points = np.array(
        [
            [x * scaled_width / FIELD_WIDTH, y * scaled_height / FIELD_HEIGHT]
            for x, y in points
        ],
        dtype=np.float32,
    )

    return points


def find_midpoint_lower_side(xywh: np.ndarray) -> np.ndarray:
    """Find the midpoint of the lower side of the bounding box. Use this to determine player positions.

    Args:
        xywh (np.ndarray): Bounding box coordinates.

    Returns:
        np.ndarray: Midpoints of the lower side of the bounding box.
    """
    midpoints = []
    for box in xywh:
        x, y, w, h = box
        # Calculate the midpoint of the lower side
        midpoints.append((x, y + h / 2))
    return np.array(midpoints)


def pairwise_distance(
    query_features: torch.Tensor, gallery_features: torch.Tensor
) -> torch.Tensor:
    """Calculate the pairwise distance between query and gallery features.

    Args:
        query_features (torch.Tensor): Query features.
        gallery_features (torch.Tensor): Gallery features

    Returns:
        torch.Tensor: Pairwise distance between query and gallery features.
    """
    number_of_query = query_features.size(0)
    number_of_gallery = gallery_features.size(0)

    dist = (
        torch.pow(query_features, 2)
        .sum(dim=1, keepdim=True)
        .expand(number_of_query, number_of_gallery)
        + torch.pow(gallery_features, 2)
        .sum(dim=1, keepdim=True)
        .expand(number_of_gallery, number_of_query)
        .t()
    )
    dist.addmm_(query_features, gallery_features.t(), beta=1, alpha=-2)

    return dist


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """
    Source: https://github.com/zhunzhong07/person-re-ranking

    Created on Mon Jun 26 14:46:56 2017
    @author: luohao
    Modified by Houjing Huang, 2017-12-22.
    - This version accepts distance matrix instead of raw features.
    - The difference of `/` division between python 2 and 3 is handled.
    - numpy.float16 is replaced by numpy.float32 for numerical precision.

    CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
    url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
    Matlab version: https://github.com/zhunzhong07/person-re-ranking

    API
    q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
    q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
    g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
    k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
    Returns:
      final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
    """

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
        [
            np.concatenate([q_q_dist, q_g_dist], axis=1),
            np.concatenate([q_g_dist.T, g_g_dist], axis=1),
        ],
        axis=0,
    )
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1.0 * original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, : k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[
                candidate, : int(np.around(k1 / 2.0)) + 1
            ]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, : int(np.around(k1 / 2.0)) + 1
            ]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(
                np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            ) > 2.0 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1.0 * weight / np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )
        jaccard_dist[i] = 1 - temp_min / (2.0 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def get_matches_from_reranked_distance_mat(distance_mat: np.ndarray) -> dict[int, int]:
    """Get the best matches from the reranked distance matrix.

    Args:
        distance_mat (np.ndarray): Distance matrix.

    Returns:
        dict[int, int]: Best matches.
    """
    best_matches = {}
    while distance_mat.min() < np.inf:
        min_indices = np.unravel_index(
            np.argmin(distance_mat, axis=None), distance_mat.shape
        )

        row, col = min_indices

        if (np.argmin(distance_mat[:, col]) == row) and (
            np.argmin(distance_mat[row, :]) == col
        ):
            best_matches[row] = col

            # Set the row and column to a very low value to ensure one-to-one matching
            distance_mat[row, :] = np.inf
            distance_mat[:, col] = np.inf
        else:
            # In case of no mutual best match, set this cell to a very low value
            distance_mat[row, col] = np.inf

    return best_matches


def crop_bbox_from_image(
    image: np.ndarray, bboxes: np.ndarray, preprocess: Callable | None = None
) -> list[torch.Tensor | np.ndarray]:
    """Crop bounding boxes from the image and preprocess them.

    Args:
        image (np.ndarray): Path to the image.
        bboxes (np.ndarray): Bounding boxes.
        preprocess (Callable): Preprocessing function.

    Returns:
        list[torch.Tensor]: List of cropped and preprocessed bounding boxes.
    """
    cropped_images = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the image to PIL format
    image = Image.fromarray(image)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cropped_image = image.crop((x1, y1, x2, y2))
        if preprocess:
            cropped_image = preprocess(cropped_image)
        if isinstance(cropped_image, torch.Tensor):
            cropped_images.append(cropped_image.unsqueeze(0))
        else:
            cropped_images.append(cropped_image)
    return cropped_images


def pad_to_square(image, fill=0) -> Image.Image:
    """
    Pad the input image to make it square.

    Args:
        image (PIL.Image): Input image.
        fill (int): Fill value for padding.

    Returns:
        PIL.Image: Padded image.
    """
    width, height = image.size
    if width == height:
        return image

    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,  # left
        (max_side - height) // 2,  # top
        (max_side - width + 1) // 2,  # right
        (max_side - height + 1) // 2,  # bottom
    )
    padded_image = transforms.functional.pad(
        image, padding, fill=fill, padding_mode="constant"
    )
    return padded_image


def check_valid_movements(
    best_matches: dict[int, int],
    previous_positions: np.ndarray,
    current_positions: np.ndarray,
):
    # TODO: Productionize this function
    for query_idx, gallery_idx in best_matches.items():
        real_distance = calculate_real_distance(
            1600, 851, previous_positions[query_idx], current_positions[gallery_idx]
        )
        if real_distance > 2.0:
            # assume athletes top speed is 10 m/s, which is 32.8 ft/s, 1 frame in a 60 fps video is 1/60 s
            # so within 1 frame, the maximum distance is 32.8/60 = 0.5467 ft
            print(
                query_idx,
                gallery_idx,
                previous_positions[query_idx],
                current_positions[gallery_idx],
                real_distance,
            )


def calculate_real_distance(
    canvas_width: int, canvas_height: int, coord1: tuple, coord2: tuple
) -> float:
    """
    Calculate the real-world distance of movement on a basketball court.

    Args:
        canvas_width (int): The width of the canvas.
        canvas_height (int): The height of the canvas.
        coord1 (tuple): The (x, y) coordinates of the player in the first frame.
        coord2 (tuple): The (x, y) coordinates of the player in the second frame.

    Returns:
        float: The real-world distance in feet.
    """

    # Real dimensions of a basketball court
    real_court_length = 94.0  # feet
    real_court_width = 50.0  # feet

    # Calculate the scale factors for x and y coordinates
    scale_x = real_court_length / canvas_width
    scale_y = real_court_width / canvas_height

    # Calculate the pixel distance between the two points
    pixel_distance = math.sqrt(
        (coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2
    )

    # Scale the pixel distance to real-world distance
    real_distance = pixel_distance * math.sqrt(scale_x**2 + scale_y**2)

    return real_distance
