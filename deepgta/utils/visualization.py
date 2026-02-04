"""
Visualization Utilities for DeepGTA

Provides drawing functions for bounding boxes, tracks,
and other visual elements.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


# Default colors for different classes (BGR format)
DEFAULT_CLASS_COLORS = {
    0: (255, 0, 0),      # Blue
    1: (0, 255, 0),      # Green
    2: (0, 0, 255),      # Red
    3: (255, 255, 0),    # Cyan (Goaltender)
    4: (255, 0, 255),    # Magenta (Player)
    5: (0, 255, 255),    # Yellow (Puck)
    6: (128, 0, 128),    # Purple (Referee)
    7: (0, 128, 255),    # Orange
    8: (255, 128, 0),    # Light Blue
    9: (128, 255, 0),    # Lime
}

# Colors for unique track IDs
TRACK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128),
    (128, 128, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255),
]


class Visualizer:
    """Visualization helper for drawing tracks and detections."""

    def __init__(
        self,
        class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        thickness: int = 2,
        font_scale: float = 0.6,
        use_track_colors: bool = True
    ):
        """Initialize the visualizer.

        Args:
            class_colors: Dictionary mapping class_id to BGR color tuple
            thickness: Line thickness for boxes
            font_scale: Font scale for labels
            use_track_colors: Use unique colors for each track ID
        """
        self.class_colors = class_colors or DEFAULT_CLASS_COLORS
        self.thickness = thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.use_track_colors = use_track_colors

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List,
        class_names: Optional[List[str]] = None,
        draw_labels: bool = True,
        draw_ids: bool = True
    ) -> np.ndarray:
        """Draw tracks on a frame.

        Args:
            frame: Input frame (BGR)
            tracks: List of track objects with tlbr, track_id, class_id attributes
            class_names: Optional list of class names
            draw_labels: Whether to draw class labels
            draw_ids: Whether to draw track IDs

        Returns:
            Frame with drawn tracks
        """
        output = frame.copy()

        for track in tracks:
            # Get bounding box
            if hasattr(track, 'tlbr'):
                bbox = track.tlbr
            elif hasattr(track, 'bbox'):
                bbox = track.bbox
            else:
                continue

            x1, y1, x2, y2 = map(int, bbox[:4])
            track_id = track.track_id if hasattr(track, 'track_id') else 0
            class_id = track.class_id if hasattr(track, 'class_id') else 0
            score = track.score if hasattr(track, 'score') else 1.0

            # Get color
            if self.use_track_colors:
                color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
            else:
                color = self.class_colors.get(class_id, (255, 255, 255))

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.thickness)

            # Draw label
            if draw_labels or draw_ids:
                label_parts = []
                if draw_ids:
                    label_parts.append(f"ID:{track_id}")
                if draw_labels and class_names and class_id < len(class_names):
                    label_parts.append(class_names[class_id])
                if draw_labels:
                    label_parts.append(f"{score:.2f}")

                label = " ".join(label_parts)
                self._draw_label(output, label, (x1, y1), color)

        return output

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List,
        class_names: Optional[List[str]] = None,
        draw_labels: bool = True
    ) -> np.ndarray:
        """Draw detections on a frame.

        Args:
            frame: Input frame (BGR)
            detections: List of Detection objects
            class_names: Optional list of class names
            draw_labels: Whether to draw labels

        Returns:
            Frame with drawn detections
        """
        output = frame.copy()

        for det in detections:
            # Get bounding box
            if hasattr(det, 'bbox'):
                bbox = det.bbox
            elif hasattr(det, 'tlbr'):
                bbox = det.tlbr
            else:
                continue

            x1, y1, x2, y2 = map(int, bbox[:4])
            class_id = det.class_id if hasattr(det, 'class_id') else 0
            confidence = det.confidence if hasattr(det, 'confidence') else 1.0

            color = self.class_colors.get(class_id, (255, 255, 255))

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.thickness)

            # Draw label
            if draw_labels:
                if class_names and class_id < len(class_names):
                    label = f"{class_names[class_id]}: {confidence:.2f}"
                else:
                    label = f"cls{class_id}: {confidence:.2f}"
                self._draw_label(output, label, (x1, y1), color)

        return output

    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int]
    ):
        """Draw a label with background.

        Args:
            frame: Frame to draw on (modified in place)
            label: Text label
            position: Top-left position (x, y)
            color: Box color (BGR)
        """
        x, y = position

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, 1
        )

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x, y - text_h - 5),
            (x + text_w + 2, y),
            color,
            -1  # Filled
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (x + 1, y - 3),
            self.font,
            self.font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    def draw_frame_info(
        self,
        frame: np.ndarray,
        frame_id: int,
        num_tracks: int,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """Draw frame information overlay.

        Args:
            frame: Input frame (BGR)
            frame_id: Current frame number
            num_tracks: Number of active tracks
            fps: Optional processing FPS

        Returns:
            Frame with info overlay
        """
        output = frame.copy()
        h, w = output.shape[:2]

        # Build info text
        info_lines = [
            f"Frame: {frame_id}",
            f"Tracks: {num_tracks}"
        ]
        if fps is not None:
            info_lines.append(f"FPS: {fps:.1f}")

        # Draw background
        y_offset = 10
        max_width = 0
        for line in info_lines:
            (text_w, text_h), _ = cv2.getTextSize(line, self.font, self.font_scale, 1)
            max_width = max(max_width, text_w)

        cv2.rectangle(
            output,
            (5, 5),
            (15 + max_width, 15 + len(info_lines) * 25),
            (0, 0, 0),
            -1
        )

        # Draw text
        for i, line in enumerate(info_lines):
            cv2.putText(
                output,
                line,
                (10, 25 + i * 25),
                self.font,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return output


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Get a unique color for a track ID.

    Args:
        track_id: Track identifier

    Returns:
        BGR color tuple
    """
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def create_grid_visualization(
    frames: List[np.ndarray],
    grid_size: Tuple[int, int] = (2, 2),
    resize: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Create a grid visualization from multiple frames.

    Args:
        frames: List of frames
        grid_size: Grid dimensions (rows, cols)
        resize: Optional resize for each frame

    Returns:
        Combined grid image
    """
    rows, cols = grid_size

    if resize:
        frames = [cv2.resize(f, resize) for f in frames]

    # Pad with black frames if needed
    h, w = frames[0].shape[:2]
    while len(frames) < rows * cols:
        frames.append(np.zeros((h, w, 3), dtype=np.uint8))

    # Create grid
    grid_rows = []
    for r in range(rows):
        row_frames = frames[r * cols:(r + 1) * cols]
        grid_rows.append(np.hstack(row_frames))

    return np.vstack(grid_rows)
