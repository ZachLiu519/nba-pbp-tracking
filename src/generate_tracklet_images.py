import argparse

import os

from .visualize import Visualizer


def main(videos_folder_path: str):
    visualizer = Visualizer(
        court_image_path="nba-pbp-tracking/src/assets/images/court.png",
        mapper_weights_path="nba-pbp-tracking/src/model_weights/best_resnet18.pth",
        tracker_weights_path="nba-pbp-tracking/src/model_weights/best_yolo.pt",
        tracker_confidence_threshold=0.9,
        tracker_overlap_threshold=0.9,
        reid_weights_path="nba-pbp-tracking/src/model_weights/best_clip.pth",
    )

    for video_name in os.listdir(videos_folder_path):

        visualizer(
            video_path=os.path.join(videos_folder_path, video_name),
            output_flag=False,  # Set to False to disable video output
            output_path="",
        )

        # Save all players' tracklets images to disk
        for player_idx, player_images in visualizer.reid.tracklet_images.items():
            for i, tracklet_image in enumerate(player_images):
                tracklet_image.save(
                    f"nba-pbp-tracking/src/videos/images/{video_name.split(".")[0]}_tracklet_images/player_{player_idx}_tracklet_{i}.png"
                )


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--videos_folder_path", type=str, required=True)
    args = argParser.parse_args()
    main(videos_folder_path=args.videos_folder_path)
