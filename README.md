# NBA Play-by-Play Video Tracking Tool

This project sets out to provide a tool that provides player identification and player court position tracking on publicly available [NBA play by play video data](https://www.nba.com/stats/help/videostatus).

## Installation

```bash
git clone https://github.com/ZachLiu519/nba-pbp-tracking.git
cd nba-play-by-play-tracking
```

Use your favorite environment management tool to create an environment, then

```bash
pip install -r requirements.txt
```

## Usage

### Tracking video generation

Download mapper weights, tracker weights and reid model weights from release. Navigate to `src/visualize.py`. Use `src/assets/images/court.png` or your favorite court image in argument `court_image_path`.

## Credits

- https://github.com/DeepSportradar/2022-winners-camera-calibration-challenge
- https://github.com/DeepSportradar/2022-winners-player-reidentification-challenge

## License

The project is licensed under the MIT License. See [LICENSE](.github/LICENSE) for more information.