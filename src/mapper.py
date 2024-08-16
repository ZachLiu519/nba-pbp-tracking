"""
Mapper instance that output homography matrix from the input NBA play by play video projected onto the court.
"""

import logging

import cv2
import numpy as np
import torch
from torchvision import transforms

from .constants import MAPPER_MODEL_INPUT
from .mapper_model.resnet import resnet18
from .utils import getFieldPoints

Logger = logging.getLogger(__name__)


class Mapper:
    def __init__(self, model_weights_path: str, court_image_path: str) -> None:
        """
        Initialize the Mapper instance with the model weights.

        Args:
            model_weights_path (str): Path to the resnet model weights.
            court_image_path (str): Path to the basketball court image.

        Returns:
            None
        """
        self.device = torch.device(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        weights = torch.load(model_weights_path, map_location=self.device)
        self.model = resnet18(weights=weights, dilation=[1, 2, 2])
        self.model.to(self.device)
        self.model.eval()

        full_court = cv2.imread(court_image_path)
        court_image_width, court_image_height = full_court.shape[1], full_court.shape[0]

        self.field_points = getFieldPoints(
            scaled_width=court_image_width, scaled_height=court_image_height
        )

        self.src_points = []
        self.dst_points = []

    def __call__(self, frame: np.ndarray):
        """
        Get the homography matrix from the input frame.

        Args:
            frame (np.ndarray): Input frame.

        Returns:
            np.ndarray: Homography matrix.
        """
        frame_height, frame_width = frame.shape[:2]

        tf = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        with torch.no_grad():
            model_output = (
                self.model(
                    tf(cv2.resize(frame, MAPPER_MODEL_INPUT))
                    .to(self.device)
                    .unsqueeze(0)
                )[0]
                .cpu()
                .numpy()
            )

            """
            Source: https://github.com/CEA-LIST/KaliCalib/blob/main/kalicalib/estimateHomography.py
            """

            pixelScores = np.swapaxes(model_output, 0, 2)
            pixelMaxScores = np.max(pixelScores, axis=2, keepdims=2)
            pixelMax = pixelScores == pixelMaxScores
            pixelMap = np.swapaxes(pixelMax, 0, 2).astype(np.uint8)
            pixelMap = (model_output > 0) * pixelMap

            for j in range(91):
                M = cv2.moments(pixelMap[j])
                # calculate x,y coordinate of center
                if M["m00"] == 0:
                    continue
                p = (M["m01"] / M["m00"], M["m10"] / M["m00"])
                p *= np.array(
                    [frame_height / 135, frame_height / 135]
                )  # scale up to original image size

                self.src_points.append(p[::-1])
                self.dst_points.append(self.field_points[j])

        if len(self.src_points) < 4 or len(self.dst_points) < 4:
            Logger.warning("Not enough points to calculate homography matrix.")
            return None

        h = cv2.findHomography(
            srcPoints=np.array(self.src_points, dtype=np.float32),
            dstPoints=np.array(self.dst_points, dtype=np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=35.0,
            maxIters=2000,
        )[0]

        return h
