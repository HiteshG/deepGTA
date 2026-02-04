# DeepGTA

**Flexible Multi-Object Tracking Pipeline**

DeepGTA is a configurable multi-object tracking (MOT) pipeline that combines:
- **YOLOv11x Detection** with custom class handling
- **Deep-EIoU Online Tracking** with ReID features
- **GTA-Link Offline Refinement** for identity correction

Based on the STC-2025 winning solution for soccer multi-object tracking.

## Features

- **Configurable Class Handling**: Track specific classes, handle special classes (e.g., single-instance objects like puck/ball)
- **Multi-Scale Detection**: Optional test-time augmentation (TTA) for improved detection
- **ReID Integration**: OSNet-based appearance features for robust association
- **Occlusion Handling**: Delayed matching for occluded objects
- **Offline Refinement**: DBSCAN-based ID switch detection and tracklet merging
- **Easy Configuration**: Single dataclass configuration for all pipeline settings

## Installation

```bash
# Clone the repository
git clone https://github.com/HiteshG/DeepGTA.git
cd DeepGTA

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

```python
from deepgta import DeepGTAConfig, DeepGTAPipeline

# Configure the pipeline
config = DeepGTAConfig(
    # Detection settings
    yolo_weights="path/to/model.pt",
    class_names=["Background", "Player", "Ball", "Referee"],
    track_classes=[1, 2, 3],     # Track Player, Ball, Referee
    special_classes=[2],          # Ball: keep only max-confidence detection
    detection_conf_thresh=0.5,

    # Tracking settings
    with_reid=True,
    reid_model="path/to/reid_model.pth",

    # Refinement settings
    use_refinement=True,

    # Output settings
    output_video=True,
    draw_tracks=True,
)

# Create pipeline
pipeline = DeepGTAPipeline(config)

# Process video
output_path, tracklets = pipeline.process_video(
    video_path="input.mp4",
    output_path="output_tracked.mp4"
)

print(f"Tracked {len(tracklets)} unique identities")
```

## Configuration Options

### Detection Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yolo_weights` | *required* | Path to YOLO model weights |
| `class_names` | *required* | List of class names |
| `track_classes` | *required* | Class indices to track |
| `special_classes` | `[]` | Classes with max-conf-only selection |
| `detection_conf_thresh` | `0.5` | Detection confidence threshold |
| `detection_iou_thresh` | `0.7` | NMS IoU threshold |
| `use_tta` | `False` | Enable multi-scale TTA |

### Tracking Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `track_high_thresh` | `0.7` | High confidence threshold |
| `track_low_thresh` | `0.4` | Low confidence threshold |
| `new_track_thresh` | `0.8` | New track initialization threshold |
| `track_buffer` | `90` | Lost track buffer (frames) |
| `match_thresh` | `0.8` | Association matching threshold |
| `with_reid` | `True` | Enable ReID features |
| `reid_model` | `None` | Path to ReID model |

### Refinement Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_refinement` | `True` | Enable GTA-Link refinement |
| `split_eps` | `0.65` | DBSCAN epsilon for splitting |
| `split_min_samples` | `15` | DBSCAN min samples |
| `split_min_len` | `50` | Min tracklet length for splitting |
| `merge_dist_thresh` | `0.35` | Merging distance threshold |

## Pipeline Architecture

```
Video Input
    │
    ▼
┌─────────────────────────────┐
│     YOLOv11x Detection      │
│  - Multi-class filtering    │
│  - Special class handling   │
│  - Per-class NMS            │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Deep-EIoU Online Tracker  │
│  - EIoU-based association   │
│  - ReID feature matching    │
│  - Kalman filter prediction │
│  - Occlusion handling       │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  GTA-Link Offline Refiner   │
│  - DBSCAN ID switch detect  │
│  - Spatial constraint merge │
│  - Feature-based connection │
└─────────────────────────────┘
    │
    ▼
Output Video + MOT Results
```

## Example: Hockey Tracking

```python
from deepgta import DeepGTAConfig, DeepGTAPipeline

config = DeepGTAConfig(
    yolo_weights="HockeyAI_model.pt",
    class_names=[
        "Center Ice", "Faceoff", "Goalpost",
        "Goaltender", "Player", "Puck", "Referee"
    ],
    track_classes=[3, 4, 5, 6],  # Goaltender, Player, Puck, Referee
    special_classes=[5],          # Puck: single instance
    detection_conf_thresh=0.5,
    detection_iou_thresh=0.7,

    with_reid=True,
    reid_model="sports_model.pth.tar-60",

    use_refinement=True,
    output_video=True,
)

pipeline = DeepGTAPipeline(config)
pipeline.process_video("hockey_game.mp4", "hockey_tracked.mp4")
```

## Colab Demo

Try DeepGTA in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HiteshG/DeepGTA/blob/main/notebooks/DeepGTA_Demo.ipynb)

## Credits

This project builds upon:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Deep-EIoU](https://github.com/MCG-NJU/Deep-EIoU)
- [GTA-Link](https://github.com/sjc042/gta-link)
- [TorchReID](https://github.com/KaiyangZhou/deep-person-reid)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use DeepGTA in your research, please cite:

```bibtex
@software{deepgta2025,
  title={DeepGTA: Flexible Multi-Object Tracking Pipeline},
  author={DeepGTA Contributors},
  year={2025},
  url={https://github.com/HiteshG/DeepGTA}
}
```
