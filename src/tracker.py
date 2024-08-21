"""
Tracker instance that tracks the basketball players on the court and returns their positions.
"""

import numpy as np
from ultralytics import YOLO

from .utils import find_midpoint_lower_side


class Tracker:
    def __init__(
        self,
        model_weights_path: str,
        confidence_threshold: float,
        overlap_threshold: float,
    ) -> None:
        """
        Initialize the Tracker instance with the model weights and confidence threshold.

        Args:
            model_weights_path (str): Path to the YOLO model weights.
            confidence_threshold (float): Confidence threshold for the YOLO model.
            overlap_threshold (float): Overlap threshold for the YOLO model.

        Returns:
            None
        """
        self.model = YOLO(model_weights_path)
        self.model.conf = confidence_threshold
        self.model.iou = overlap_threshold
        self.model_results = None

        # Cache to store the positions of the basketball players.
        self.positions_cache = {}

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        Return the position of the basketball players in the input frame.

        Args:
            frame (np.ndarray): Input frame.

        Returns:
            np.ndarray: Coordinates of the basketball players in the frame.
        """
        self.model_results = self.model(frame)[0]
        boxes = self.model_results.boxes
        positions = find_midpoint_lower_side(xywh=boxes.xywh.cpu().numpy())

        return positions

    def set_positions_cache(self, positions: np.ndarray) -> None:
        """
        Set the positions cache with the input positions.

        Args:
            positions (np.ndarray): Input positions.

        Returns:
            None
        """
        for idx, position in enumerate(positions):
            self.positions_cache[idx] = position

    def update_positions_cache(
        self, best_matches: dict[int, int], new_positions: np.ndarray
    ) -> None:
        """
        Update the positions cache with the best matches and new positions.

        Args:
            best_matches (dict[int, int]): Best matches between query and gallery indices.
            new_positions (np.ndarray): New positions.

        Returns:
            None
        """
        for query_idx, gallery_idx in best_matches.items():
            self.positions_cache[query_idx] = new_positions[gallery_idx]

        # Same as rule for updating tracklets in reid.py
        if (
            len(new_positions) > len(self.positions_cache)
            and len(self.positions_cache) < 10
        ):
            for i in range(len(new_positions)):
                if i not in best_matches.values():
                    self.positions_cache[i] = new_positions[i]
