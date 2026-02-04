"""
DeepGTA Configuration Module

Provides configuration dataclass for the entire MOT pipeline including
detection, tracking, and refinement stages.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class DeepGTAConfig:
    """Configuration for the DeepGTA multi-object tracking pipeline.

    Attributes:
        yolo_weights: Path to YOLO model weights (.pt file)
        class_names: List of class names from the YOLO model
        track_classes: Class indices to track
        special_classes: Class indices where only max-confidence detection is kept
    """

    # === Detection Settings ===
    yolo_weights: str                        # Path to .pt weights
    class_names: List[str]                   # Class name list
    track_classes: List[int]                 # Class indices to track
    special_classes: List[int] = field(default_factory=list)  # Max-conf only
    detection_conf_thresh: float = 0.5
    detection_iou_thresh: float = 0.7
    detection_imgsz: List[int] = field(default_factory=lambda: [1280])
    use_tta: bool = False                    # Multi-scale TTA

    # === Tracking Settings ===
    track_high_thresh: float = 0.7           # High confidence detection threshold
    track_low_thresh: float = 0.4            # Low confidence detection threshold
    new_track_thresh: float = 0.8            # New track initialization threshold
    track_buffer: int = 90                   # Lost track buffer (frames)
    match_thresh: float = 0.8                # Matching threshold
    proximity_thresh: float = 0.5            # Distance threshold for association
    appearance_thresh: float = 0.25          # ReID similarity threshold
    with_reid: bool = True                   # Use ReID features
    reid_model: Optional[str] = None         # Path to ReID model weights

    # === Occlusion Detection Settings ===
    occ_iou_thresh: float = 0.5              # IoU threshold for occlusion detection
    occ_sim_thresh: float = 0.7              # Similarity threshold for occlusion
    occ_buffer_frames: int = 48              # Frames to buffer for occlusion handling

    # === GTA-Link Refinement Settings ===
    use_refinement: bool = True              # Enable offline refinement
    split_eps: float = 0.65                  # DBSCAN epsilon for splitting
    split_min_samples: int = 15              # DBSCAN min samples
    split_max_k: int = 2                     # Max clusters for splitting
    split_min_len: int = 50                  # Min tracklet length for splitting
    merge_dist_thresh: float = 0.35          # Connection distance threshold
    spatial_factor: float = 1.5              # Spatial constraint factor

    # === Output Settings ===
    output_video: bool = True                # Generate output video
    output_codec: str = "mp4v"               # Video codec
    compress_quality: int = 23               # FFmpeg CRF (lower=better)
    draw_tracks: bool = True                 # Draw tracking boxes
    draw_labels: bool = True                 # Draw class labels
    save_mot_txt: bool = False               # Save MOT format results
    mot_output_path: Optional[str] = None    # Path for MOT results

    # === Visualization Settings ===
    track_colors: Optional[Dict[int, Tuple[int, int, int]]] = None  # Per-class colors
    track_thickness: int = 2                 # Box line thickness
    label_font_scale: float = 0.6            # Label font size

    # === Performance Settings ===
    device: str = "cuda"                     # cuda or cpu
    batch_size: int = 1                      # Detection batch size
    frame_rate: int = 30                     # Video frame rate

    # === Debug Settings ===
    verbose: bool = False                    # Enable verbose logging
    save_intermediate: bool = False          # Save intermediate results

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.yolo_weights:
            raise ValueError("yolo_weights must be specified")
        if not self.class_names:
            raise ValueError("class_names must be specified")
        if not self.track_classes:
            raise ValueError("track_classes must be specified")

        # Validate track_classes are valid indices
        for cls_idx in self.track_classes:
            if cls_idx < 0 or cls_idx >= len(self.class_names):
                raise ValueError(f"Invalid track_class index: {cls_idx}")

        # Validate special_classes are in track_classes
        for cls_idx in self.special_classes:
            if cls_idx not in self.track_classes:
                raise ValueError(f"special_class {cls_idx} must be in track_classes")

    def get_class_name(self, class_idx: int) -> str:
        """Get class name by index."""
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return f"class_{class_idx}"

    def is_special_class(self, class_idx: int) -> bool:
        """Check if class requires special handling (max-conf only)."""
        return class_idx in self.special_classes

    def is_tracked_class(self, class_idx: int) -> bool:
        """Check if class should be tracked."""
        return class_idx in self.track_classes
