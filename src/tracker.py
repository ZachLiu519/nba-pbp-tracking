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
