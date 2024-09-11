import argparse

from .visualize import Visualizer


def main(video_path: str):
    visualizer = Visualizer(
        court_image_path="nba-pbp-tracking/src/assets/images/court.png",
        mapper_weights_path="nba-pbp-tracking/src/model_weights/best_resnet18.pth",
        tracker_weights_path="nba-pbp-tracking/src/model_weights/best_yolo.pt",
        tracker_confidence_threshold=0.9,
        tracker_overlap_threshold=0.9,
        reid_weights_path="nba-pbp-tracking/src/model_weights/best_clip.pth",
    )

    visualizer(
        video_path=video_path,
        output_flag=False,  # Set to False to disable video output
        output_path="",
    )

    video_name = video_path.split("/")[-1].split(".")[0]

    # Save all players' tracklets images to disk
    for player_idx, player_images in visualizer.reid.tracklet_images.items():
        for i, tracklet_image in enumerate(player_images):
            tracklet_image.save(
                f"nba-pbp-tracking/src/assets/images/{video_name}_tracklet_images/player_{player_idx}_tracklet_{i}.png"
            )


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--video_path", type=str, required=True)
    args = argParser.parse_args()
    main(video_path=args.video_path)
