"""
Tracking Testing Module for DeepGTA

Use this module to verify that tracking is working correctly
on your video before running the full pipeline.
"""

import cv2
import numpy as np
from typing import Optional

from .config import DeepGTAConfig
from .detection import YOLODetector
from .tracking import DeepEIoUTracker


def test_tracking_on_video(
    config: DeepGTAConfig,
    video_path: str,
    output_path: Optional[str] = None,
    max_frames: int = 100,
    show: bool = False
):
    """Test tracking on a video.

    Args:
        config: DeepGTAConfig instance
        video_path: Path to video file
        output_path: Optional path to save annotated video
        max_frames: Maximum number of frames to process
        show: Whether to display the video (requires GUI)

    Returns:
        Dictionary with tracking statistics
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Processing up to {max_frames} frames")

    # Initialize detector and tracker
    detector = YOLODetector(config)
    tracker = DeepEIoUTracker(config, frame_rate=fps)

    # Video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Statistics
    stats = {
        'frames_processed': 0,
        'total_detections': 0,
        'total_tracks': 0,
        'unique_track_ids': set(),
        'tracks_per_frame': []
    }

    frame_id = 0
    while frame_id < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Detection
        detections = detector.detect(frame)
        stats['total_detections'] += len(detections)

        # Tracking
        tracks = tracker.update(detections, None)
        stats['total_tracks'] += len(tracks)
        stats['tracks_per_frame'].append(len(tracks))

        for track in tracks:
            stats['unique_track_ids'].add(track.track_id)

        # Draw results
        annotated_frame = frame.copy()

        # Draw detections (light gray, dashed)
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (128, 128, 128), 1)

        # Draw tracks (colored, solid)
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0)
        ]

        for track in tracks:
            # Use last_tlwh for actual detection position
            bbox = track.last_tlwh
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

            color = colors[track.track_id % len(colors)]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"ID:{track.track_id}"
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Frame info
        info = f"Frame: {frame_id} | Dets: {len(detections)} | Tracks: {len(tracks)}"
        cv2.putText(annotated_frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if writer:
            writer.write(annotated_frame)

        if show:
            cv2.imshow("Tracking Test", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_id % 10 == 0:
            print(f"Frame {frame_id}: {len(detections)} detections, {len(tracks)} tracks")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    # Calculate statistics
    stats['frames_processed'] = frame_id
    stats['unique_track_ids'] = len(stats['unique_track_ids'])
    stats['avg_detections_per_frame'] = stats['total_detections'] / max(1, frame_id)
    stats['avg_tracks_per_frame'] = stats['total_tracks'] / max(1, frame_id)

    print(f"\n{'='*50}")
    print("TRACKING STATISTICS")
    print('='*50)
    print(f"Frames processed: {stats['frames_processed']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Avg detections/frame: {stats['avg_detections_per_frame']:.1f}")
    print(f"Total track instances: {stats['total_tracks']}")
    print(f"Avg tracks/frame: {stats['avg_tracks_per_frame']:.1f}")
    print(f"Unique track IDs: {stats['unique_track_ids']}")

    if output_path:
        print(f"\nOutput saved to: {output_path}")

    return stats


def quick_tracking_test(
    yolo_weights: str,
    video_path: str,
    class_names: list,
    track_classes: list,
    output_path: Optional[str] = None,
    max_frames: int = 100
):
    """Quick tracking test with minimal configuration.

    Args:
        yolo_weights: Path to YOLO weights
        video_path: Path to video
        class_names: List of class names
        track_classes: Class indices to track
        output_path: Optional output video path
        max_frames: Maximum frames to process
    """
    config = DeepGTAConfig(
        yolo_weights=yolo_weights,
        class_names=class_names,
        track_classes=track_classes,
        detection_conf_thresh=0.5,
        with_reid=False,  # Disable ReID for quick test
        verbose=False
    )

    test_tracking_on_video(config, video_path, output_path, max_frames, show=False)


def verify_matching():
    """Verify that the matching functions work correctly."""
    from .tracking import matching

    print("Testing expand_box function...")

    # Test box in [x1, y1, x2, y2] format
    box = [100, 200, 150, 300]  # x1=100, y1=200, x2=150, y2=300
    # Width = 50, Height = 100

    # Expand by factor 0.5
    expanded = matching.expand_box(box, 0.5)

    # Expected:
    # expand_w = 2 * 50 * 0.5 + 50 = 100
    # expand_h = 2 * 100 * 0.5 + 100 = 200
    # new_x1 = 100 - 50 = 50
    # new_y1 = 200 - 100 = 100
    # new_x2 = 150 + 50 = 200
    # new_y2 = 300 + 100 = 400

    expected = [50, 100, 200, 400]

    print(f"Original box: {box}")
    print(f"Expanded box: {expanded}")
    print(f"Expected:     {expected}")

    # Check if close enough (allow small float differences)
    is_correct = all(abs(a - b) < 0.001 for a, b in zip(expanded, expected))
    print(f"expand_box test: {'PASS' if is_correct else 'FAIL'}")

    # Test IoU
    print("\nTesting IoU calculation...")
    box1 = np.array([[0, 0, 10, 10]])  # 10x10 box
    box2 = np.array([[5, 5, 15, 15]])  # 10x10 box, overlapping 5x5

    iou_matrix = matching.ious(box1, box2)
    # Intersection: 5*5 = 25
    # Union: 10*10 + 10*10 - 25 = 175
    # IoU: 25/175 ≈ 0.143
    expected_iou = 25 / 175

    print(f"Box1: {box1[0]}")
    print(f"Box2: {box2[0]}")
    print(f"Computed IoU: {iou_matrix[0, 0]:.4f}")
    print(f"Expected IoU: {expected_iou:.4f}")

    is_iou_correct = abs(iou_matrix[0, 0] - expected_iou) < 0.001
    print(f"IoU test: {'PASS' if is_iou_correct else 'FAIL'}")

    return is_correct and is_iou_correct


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Running matching verification tests...")
        verify_matching()
    elif len(sys.argv) < 4:
        print("Usage: python -m deepgta.test_tracking <yolo_weights> <video_path> <class_names_comma_separated>")
        print("Example: python -m deepgta.test_tracking model.pt video.mp4 'person,car,truck'")
        sys.exit(1)
    else:
        weights = sys.argv[1]
        video = sys.argv[2]
        classes = sys.argv[3].split(',')
        track_all = list(range(len(classes)))

        quick_tracking_test(weights, video, classes, track_all)
