"""
Visualizer instance to plot the locations of the basketball players on the court and track their identities.
"""

from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
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

        self.clip_model = create_model_from_pretrained("ViT-B/16", "openai")
        self.tokenizer = get_tokenizer("ViT-B-16")

        self.projected_positions = []
        self.gallery_features = defaultdict(list)
        self.player_jersey_number_color_map = {}

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
                bboxes=self.tracker.model_results.boxes.xyxy[
                    torch.where(
                        self.tracker.model_results.boxes.cls == 0
                    )  # 0 is the class label for players
                ]
                .cpu()
                .numpy(),
                preprocess=self.reid_preprocess,
            )

            if i == 0:
                self.reid.setup_tracklet(images=detected_player_images)
                self.tracker.set_positions_cache(positions=positions)
            else:
                position_distance_matrix = self.tracker.get_position_distance_matrix(
                    current_positions=positions
                )
                best_matches, gallery_features = self.reid.reidentify(
                    images=detected_player_images,
                    position_distance_matrix=position_distance_matrix,
                )
                self.reid.update_tracklet(
                    best_matches=best_matches, gallery_features=gallery_features
                )
                self.tracker.update_positions_cache(
                    best_matches=best_matches, new_positions=positions
                )

            if i == 0:
                self.projected_positions.append(
                    {idx: point for idx, point in enumerate(projected_positions)}
                )
            else:
                gallery_to_query = {v: k for k, v in best_matches.items()}
                self.projected_positions.append(
                    {
                        gallery_to_query[idx]: point
                        for idx, point in enumerate(projected_positions)
                    }
                )

            for idx, gallery_feature in enumerate(gallery_features):
                self.gallery_features[idx].append(gallery_feature)

            # for idx, point in enumerate(projected_positions):
            #     if i == 0:
            #         cv2.circle(
            #             full_court_image_to_plot,
            #             tuple(map(int, point)),
            #             10,
            #             (0, 255, 0),
            #             -1,
            #         )
            #         cv2.putText(
            #             full_court_image_to_plot,
            #             str(idx),
            #             (int(point[0]) + 10, int(point[1]) - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             1,
            #             (0, 128, 255),
            #             2,
            #             cv2.LINE_AA,
            #         )
            #     else:
            #         gallery_to_query = {v: k for k, v in best_matches.items()}
            #         if idx in gallery_to_query:
            #             cv2.circle(
            #                 full_court_image_to_plot,
            #                 tuple(map(int, point)),
            #                 10,
            #                 (0, 255, 0),
            #                 -1,
            #             )
            #             cv2.putText(
            #                 full_court_image_to_plot,
            #                 str(gallery_to_query[idx]),
            #                 (int(point[0]) + 10, int(point[1]) - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX,
            #                 1,
            #                 (0, 128, 255),
            #                 2,
            #                 cv2.LINE_AA,
            #             )
            # out.write(full_court_image_to_plot)

        cap.release()
        out.release()

    def get_jersey_number_color(self):
        colors = [
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "grey",
            "orange",
        ]
        number_text = self.tokenizer(
            [f"a basketball player with jersey number {i}" for i in range(100)]
        )
        color_text = self.tokenizer([f"a {i} jersey, color {i}" for i in colors])

        with torch.no_grad(), torch.cuda.amp.autocast():
            number_text_features = self.clip_model.encode_text(number_text)
            number_text_features /= number_text_features.norm(dim=-1, keepdim=True)

            color_text_features = self.clip_model.encode_text(color_text)
            color_text_features /= color_text_features.norm(dim=-1, keepdim=True)

            for idx, gallery_features in self.gallery_features.items():
                number_labels = []
                color_labels = []
                gallery_features = torch.stack(gallery_features)
                gallery_features = self.clip_model.encode_image(gallery_features)
                for gallery_feature in gallery_features:
                    gallery_feature /= gallery_feature.norm(dim=-1, keepdim=True)

                    number_text_probs = (
                        100.0 * gallery_feature @ number_text_features.T
                    ).softmax(dim=-1)

                    color_text_probs = (
                        100.0 * gallery_feature @ color_text_features.T
                    ).softmax(dim=-1)

                    if number_text_probs.var() > 1e-4:
                        number_labels.append(number_text_probs.argmax())

                    color_labels.append(color_text_probs.argmax())

                number_labels_counter = Counter(number_labels)
                color_labels_counter = Counter(color_labels)

                self.player_jersey_number_color_map[idx] = (
                    max(number_labels_counter, key=number_labels_counter.get),
                    max(color_labels_counter, key=color_labels_counter.get),
                )
