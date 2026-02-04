"""
Detection Testing Module for DeepGTA

Use this module to verify that detection is working correctly
on your video before running the full pipeline.
"""

import cv2
import numpy as np
from typing import Optional

from .config import DeepGTAConfig
from .detection import YOLODetector


def test_detection_on_frame(
    config: DeepGTAConfig,
    video_path: str,
    frame_number: int = 1,
    save_path: Optional[str] = None,
    show: bool = True
):
    """Test detection on a specific frame.

    Args:
        config: DeepGTAConfig instance
        video_path: Path to video file
        frame_number: Frame number to test (1-indexed)
        save_path: Optional path to save annotated frame
        show: Whether to display the frame (requires GUI)

    Returns:
        Tuple of (frame, detections, raw_detections)
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {frame_number}")

    print(f"Frame shape: {frame.shape}")
    print(f"Frame dtype: {frame.dtype}")

    # Initialize detector
    detector = YOLODetector(config)

    # Get raw detections (before class filtering)
    print("\n=== Running YOLO Detection ===")
    raw_detections = detector.detect(frame, filter_classes=False)
    print(f"\nRaw detections (all classes): {len(raw_detections)}")

    for i, det in enumerate(raw_detections):
        print(f"  [{i}] class={det.class_id} ({det.class_name}), "
              f"conf={det.confidence:.3f}, "
              f"bbox=[{det.bbox[0]:.1f}, {det.bbox[1]:.1f}, {det.bbox[2]:.1f}, {det.bbox[3]:.1f}]")

    # Get filtered detections
    filtered_detections = detector.detect(frame, filter_classes=True)
    print(f"\nFiltered detections (track_classes={config.track_classes}): {len(filtered_detections)}")

    for i, det in enumerate(filtered_detections):
        print(f"  [{i}] class={det.class_id} ({det.class_name}), "
              f"conf={det.confidence:.3f}, "
              f"bbox=[{det.bbox[0]:.1f}, {det.bbox[1]:.1f}, {det.bbox[2]:.1f}, {det.bbox[3]:.1f}]")

    # Draw detections on frame
    annotated_frame = frame.copy()

    # Define colors for different classes
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
    ]

    for det in raw_detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        color = colors[det.class_id % len(colors)]

        # Use dashed line for non-tracked classes
        if det.class_id in config.track_classes:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        else:
            # Draw dashed rectangle for non-tracked
            for i in range(0, x2 - x1, 10):
                cv2.line(annotated_frame, (x1 + i, y1), (min(x1 + i + 5, x2), y1), color, 1)
                cv2.line(annotated_frame, (x1 + i, y2), (min(x1 + i + 5, x2), y2), color, 1)
            for i in range(0, y2 - y1, 10):
                cv2.line(annotated_frame, (x1, y1 + i), (x1, min(y1 + i + 5, y2)), color, 1)
                cv2.line(annotated_frame, (x2, y1 + i), (x2, min(y1 + i + 5, y2)), color, 1)

        # Label
        label = f"{det.class_name}: {det.confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_frame, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add info text
    info_text = f"Frame: {frame_number} | Raw: {len(raw_detections)} | Filtered: {len(filtered_detections)}"
    cv2.putText(annotated_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save if requested
    if save_path:
        cv2.imwrite(save_path, annotated_frame)
        print(f"\nSaved annotated frame to: {save_path}")

    # Show if requested
    if show:
        try:
            cv2.imshow("Detection Test", annotated_frame)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Could not display image (no GUI?): {e}")

    return frame, filtered_detections, raw_detections


def test_detection_multiple_frames(
    config: DeepGTAConfig,
    video_path: str,
    num_frames: int = 5,
    save_dir: Optional[str] = None
):
    """Test detection on multiple evenly-spaced frames.

    Args:
        config: DeepGTAConfig instance
        video_path: Path to video file
        num_frames: Number of frames to test
        save_dir: Optional directory to save annotated frames
    """
    import os

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video has {total_frames} frames")
    print(f"Testing {num_frames} evenly-spaced frames\n")

    # Get evenly spaced frame numbers
    frame_numbers = [
        int(i * total_frames / (num_frames + 1))
        for i in range(1, num_frames + 1)
    ]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    total_raw = 0
    total_filtered = 0

    for frame_num in frame_numbers:
        print(f"\n{'='*50}")
        print(f"Testing Frame {frame_num}")
        print('='*50)

        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"frame_{frame_num:06d}.jpg")

        _, filtered, raw = test_detection_on_frame(
            config, video_path, frame_num, save_path=save_path, show=False
        )

        total_raw += len(raw)
        total_filtered += len(filtered)

    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    print(f"Frames tested: {num_frames}")
    print(f"Total raw detections: {total_raw}")
    print(f"Total filtered detections: {total_filtered}")
    print(f"Avg raw per frame: {total_raw / num_frames:.1f}")
    print(f"Avg filtered per frame: {total_filtered / num_frames:.1f}")


def quick_test(
    yolo_weights: str,
    video_path: str,
    class_names: list,
    track_classes: list,
    frame_number: int = 1,
    conf_thresh: float = 0.5
):
    """Quick test with minimal configuration.

    Args:
        yolo_weights: Path to YOLO weights
        video_path: Path to video
        class_names: List of class names
        track_classes: Class indices to track
        frame_number: Frame to test
        conf_thresh: Confidence threshold
    """
    config = DeepGTAConfig(
        yolo_weights=yolo_weights,
        class_names=class_names,
        track_classes=track_classes,
        detection_conf_thresh=conf_thresh,
        verbose=True
    )

    test_detection_on_frame(config, video_path, frame_number, show=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python -m deepgta.test_detection <yolo_weights> <video_path> <class_names_comma_separated>")
        print("Example: python -m deepgta.test_detection model.pt video.mp4 'person,car,truck'")
        sys.exit(1)

    weights = sys.argv[1]
    video = sys.argv[2]
    classes = sys.argv[3].split(',')

    # By default, track all classes
    track_all = list(range(len(classes)))

    quick_test(weights, video, classes, track_all)
