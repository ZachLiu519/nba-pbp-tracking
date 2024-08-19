"""
ReID instance to re identify basketball players detected between video frames.
"""

import torch

from .reid_model.image_clip import ImageClip
from .utils import get_matches_from_reranked_distance_mat, pairwise_distance, re_ranking


class ReID:
    def __init__(
        self,
        model_weights_path: str,
        model_name: str = "ViT-B/16",
        pretrained: str | None = None,
    ) -> None:
        """
        Initialize the ReID instance with the model weights.

        Args:
            model_weights_path (str): Path to the ReID model weights.

        Returns:
            None
        """
        self.device = torch.device(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        weights = torch.load(model_weights_path, map_location=self.device)
        self.model = ImageClip(model_name=model_name, pretrained=pretrained)
        self.model.load_state_dict(state_dict=weights)
        self.model.to(self.device)

        # Initialize tracklets, a dictionary to store the features of detected basketball players.
        self.tracklets: dict[int, torch.Tensor] = {}

    def setup_tracklet(self, images: list[torch.Tensor]) -> None:
        """
        Setup tracklets for each image of detected basketball players.

        Args:
            images (list[torch.Tensor]): List of image of detected basketball players.

        Returns:
            None
        """
        with torch.no_grad():
            for i, img in enumerate(images):
                img = img.to(self.device)
                self.tracklets[i] = self.model(img).detach().cpu()

    def add_tracklet(self, image: torch.Tensor) -> None:
        """
        Add a new tracklet for features of a new image of detected basketball players that didn't appear in previous detections.

        Args:
            image (torch.Tensor): Image of a newly detected basketball players.

        Returns:
            None
        """
        with torch.no_grad():
            image = image.to(self.device)
            self.tracklets[len(self.tracklets)] = self.model(image).detach().cpu()

    def update_tracklet(self, best_matches: dict[int, int], images: list[torch.Tensor]):
        """
        Given images, which are usually detections in the next frame, update the tracklets.

        Args:
            best_matches (dict[int, int]): Best matches. Key is the index of the query image and value is the index of the gallery image.
            images (list[torch.Tensor]): List of image of detected basketball players. Also known as gallery images.

        Returns:
            None
        """
        for query_idx, gallery_idx in best_matches.items():
            self.tracklets[query_idx] = (
                self.model(images[gallery_idx].to(self.device)).detach().cpu()
            )

        # check for edge cases
        # When there are fewer detections in the current frame than the previous frame,
        # i.e. len(images) < len(self.tracklets)
        # Do nothing because we will just not update the tracklet that didn't receive a match.

        # When there are more detections in the current frame than the previous frame,
        # i.e. len(images) > len(self.tracklets)
        # Add the new detections as tracklets.
        if len(images) > len(self.tracklets) and len(self.tracklets) < 10:
            for i in range(len(images)):
                if i not in best_matches.values():
                    self.add_tracklet(images[i])

    def reidentify(self, images: list[torch.Tensor]) -> dict[int, int]:
        """
        Reidentify basketball players detected between video frames.

        Args:
            images (list[torch.Tensor]): List of image of detected basketball players. Also known as gallery images.

        Returns:
            dict[int, int]: Best matches. Key is the index of the query image and value is the index of the gallery image.
        """
        images_features = []
        with torch.no_grad():
            for img in images:
                img = img.to(self.device)
                feature = self.model(img).detach().cpu()
                images_features.append(feature)
        distance_mat_q_g = pairwise_distance(
            query_features=list(self.tracklets.values()),
            gallery_features=images_features,
        )
        distance_mat_q_q = pairwise_distance(
            query_features=list(self.tracklets.values()),
            gallery_features=list(self.tracklets.values()),
        )
        distance_mat_g_g = pairwise_distance(
            query_features=images_features, gallery_features=images_features
        )
        distance_matrix_reranked = re_ranking(
            q_g_dist=distance_mat_q_g,
            q_q_dist=distance_mat_q_q,
            g_g_dist=distance_mat_g_g,
        )

        best_matches = get_matches_from_reranked_distance_mat(distance_matrix_reranked)

        return best_matches
