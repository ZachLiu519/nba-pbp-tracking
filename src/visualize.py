"""
Visualizer instance to plot the locations of the basketball players on the court and track their identities.
"""

import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm

from .mapper import Mapper
from .reid import ReID
from .tracker import Tracker
from .utils import crop_bbox_from_image, pad_to_square


class Visualizer:
    def __init__(
        self,
        court_image_path: str,
        mapper_weights_path: str,
        tracker_weights_path: str,
        tracker_confidence_threshold: float,
        tracker_overlap_threshold: float,
        reid_weights_path: str,
    ) -> None:
        """
        Initialize the mapper, tracker, and reid instances with the model weights.

        Args:
            court_image_path (str): Path to the basketball court image.
            video_path (str): Path to the NBA play by play video.
            mapper_weights_path (str): Path to the mapper model weights.
            tracker_weights_path (str): Path to the tracker model weights.
            reid_weights_path (str): Path to the reid model weights.

        Returns:
            None
        """
        self.court_image_path = court_image_path
        self.mapper_weights_path = mapper_weights_path
        self.tracker_weights_path = tracker_weights_path
        self.reid_weights_path = reid_weights_path

        self.mapper = Mapper(
            model_weights_path=mapper_weights_path, court_image_path=court_image_path
        )
        self.tracker = Tracker(
            model_weights_path=tracker_weights_path,
            confidence_threshold=tracker_confidence_threshold,
            overlap_threshold=tracker_overlap_threshold,
        )
        self.reid = ReID(model_weights_path=reid_weights_path)

        self.full_court_image = cv2.imread(court_image_path)

        self.reid_preprocess = transforms.Compose(
            [
                transforms.Lambda(pad_to_square),
                transforms.Resize(
                    size=224,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    max_size=None,
                    antialias=True,
                ),
                lambda x: x.convert("RGB"),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __call__(self, video_path: str, output_path: str) -> None:
        """
        Plot the locations of the basketball players on the court and track their identities, then save the output video.

        Args:
            video_path (str): Path to the NBA play by play video.
            output_path (str): Path to save the output video.

        Returns:
            None
        """
        # Initialize the video capture
        cap = cv2.VideoCapture(video_path)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            frame_rate,
            (self.full_court_image.shape[1], self.full_court_image.shape[0]),
        )

        for i in tqdm(range(num_frames)):
            full_court_image_to_plot = self.full_court_image.copy()

            ret, frame = cap.read()
            if not ret:
                break

            homography_matrix = self.mapper(frame)
            positions = self.tracker(frame)
            projected_positions = cv2.perspectiveTransform(
                np.array([positions], dtype=np.float32), homography_matrix
            )[0]
            detected_player_images = crop_bbox_from_image(
                image=frame,
                bboxes=self.tracker.model(frame)[0].boxes.xywh.cpu().numpy(),
                preprocess=self.reid_preprocess,
            )

            if i == 0:
                self.reid.setup_tracklet(images=detected_player_images)
            else:
                best_matches, gallery_features = self.reid.reidentify(
                    images=detected_player_images
                )
                self.reid.update_tracklet(
                    best_matches=best_matches, gallery_features=gallery_features
                )

            for idx, point in enumerate(projected_positions):
                if i == 0:
                    cv2.circle(
                        full_court_image_to_plot,
                        tuple(map(int, point)),
                        10,
                        (0, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        full_court_image_to_plot,
                        str(idx),
                        (int(point[0]) + 10, int(point[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 128, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    gallery_to_query = {v: k for k, v in best_matches.items()}
                    if idx in gallery_to_query:
                        cv2.circle(
                            full_court_image_to_plot,
                            tuple(map(int, point)),
                            10,
                            (0, 255, 0),
                            -1,
                        )
                        cv2.putText(
                            full_court_image_to_plot,
                            str(gallery_to_query[idx]),
                            (int(point[0]) + 10, int(point[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 128, 255),
                            2,
                            cv2.LINE_AA,
                        )

            out.write(full_court_image_to_plot)

        cap.release()
        out.release()
