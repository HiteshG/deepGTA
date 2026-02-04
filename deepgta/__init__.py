"""
DeepGTA - Flexible Multi-Object Tracking Pipeline

A configurable MOT pipeline combining:
- YOLOv11x detection with custom class handling
- Deep-EIoU online tracking with ReID features
- GTA-Link offline refinement

Based on the STC-2025 winning solution.
"""

__version__ = "0.1.0"
__author__ = "DeepGTA Contributors"

from .config import DeepGTAConfig
from .pipeline import DeepGTAPipeline
from .detection import YOLODetector, Detection
from .tracking import DeepEIoUTracker, STrack, BaseTrack, TrackState
from .refinement import GTALinkRefiner, Tracklet
from .reid import ReIDExtractor
from .utils import VideoReader, VideoWriter, Visualizer
from .test_detection import test_detection_on_frame, test_detection_multiple_frames, quick_test
from .test_tracking import test_tracking_on_video, quick_tracking_test, verify_matching

__all__ = [
    # Main classes
    "DeepGTAConfig",
    "DeepGTAPipeline",
    # Detection
    "YOLODetector",
    "Detection",
    # Tracking
    "DeepEIoUTracker",
    "STrack",
    "BaseTrack",
    "TrackState",
    # Refinement
    "GTALinkRefiner",
    "Tracklet",
    # ReID
    "ReIDExtractor",
    # Utils
    "VideoReader",
    "VideoWriter",
    "Visualizer",
    # Testing - Detection
    "test_detection_on_frame",
    "test_detection_multiple_frames",
    "quick_test",
    # Testing - Tracking
    "test_tracking_on_video",
    "quick_tracking_test",
    "verify_matching",
]
