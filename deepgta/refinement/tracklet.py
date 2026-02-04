"""
Tracklet Data Structure for DeepGTA

Provides the Tracklet class for storing track information including
frames, bounding boxes, scores, and ReID features.
"""

from typing import List, Optional
import numpy as np


class Tracklet:
    """Container for tracklet data.

    A tracklet represents a sequence of detections belonging to the same
    object across multiple frames.

    Attributes:
        track_id: Unique identifier for the track
        parent_id: Original track ID (for split tracklets)
        times: List of frame numbers
        scores: List of detection scores
        bboxes: List of bounding boxes [x, y, w, h]
        features: List of ReID feature vectors
        class_id: Class index of the tracked object
    """

    def __init__(
        self,
        track_id: int = None,
        frames: Optional[List[int]] = None,
        scores: Optional[List[float]] = None,
        bboxes: Optional[List[List[float]]] = None,
        feats: Optional[List[np.ndarray]] = None,
        class_id: int = 0
    ):
        """Initialize a Tracklet.

        Args:
            track_id: Unique track identifier
            frames: Frame numbers (can be single int or list)
            scores: Detection scores (can be single float or list)
            bboxes: Bounding boxes (can be single box or list of boxes)
            feats: ReID features
            class_id: Class index
        """
        self.track_id = track_id
        self.parent_id = track_id
        self.class_id = class_id

        # Handle single values or lists
        self.scores = scores if isinstance(scores, list) else [scores] if scores is not None else []
        self.times = frames if isinstance(frames, list) else [frames] if frames is not None else []
        self.bboxes = bboxes if isinstance(bboxes, list) and bboxes and isinstance(bboxes[0], list) else [bboxes] if bboxes is not None else []
        self.features = feats if feats is not None else []

    def append_det(self, frame: int, score: float, bbox: List[float]):
        """Append a detection to the tracklet.

        Args:
            frame: Frame number
            score: Detection score
            bbox: Bounding box [x, y, w, h]
        """
        self.scores.append(score)
        self.times.append(frame)
        self.bboxes.append(bbox)

    def append_feat(self, feat: np.ndarray):
        """Append a feature vector.

        Args:
            feat: Feature vector (should be normalized)
        """
        self.features.append(feat)

    def extract(self, start: int, end: int) -> 'Tracklet':
        """Extract a sub-tracklet between indices.

        Args:
            start: Start index (inclusive)
            end: End index (inclusive)

        Returns:
            New Tracklet object with extracted data
        """
        subtrack = Tracklet(
            track_id=self.track_id,
            frames=self.times[start:end + 1],
            scores=self.scores[start:end + 1],
            bboxes=self.bboxes[start:end + 1],
            feats=self.features[start:end + 1] if self.features else None,
            class_id=self.class_id
        )
        subtrack.parent_id = self.parent_id
        return subtrack

    def merge(self, other: 'Tracklet'):
        """Merge another tracklet into this one.

        Args:
            other: Tracklet to merge
        """
        self.times.extend(other.times)
        self.scores.extend(other.scores)
        self.bboxes.extend(other.bboxes)
        self.features.extend(other.features)

    @property
    def length(self) -> int:
        """Get the length of the tracklet."""
        return len(self.times)

    @property
    def start_frame(self) -> int:
        """Get the starting frame."""
        return min(self.times) if self.times else 0

    @property
    def end_frame(self) -> int:
        """Get the ending frame."""
        return max(self.times) if self.times else 0

    @property
    def avg_score(self) -> float:
        """Get average detection score."""
        return np.mean(self.scores) if self.scores else 0.0

    @property
    def mean_feature(self) -> Optional[np.ndarray]:
        """Get mean feature vector."""
        if not self.features:
            return None
        mean_feat = np.mean(self.features, axis=0)
        return mean_feat / np.linalg.norm(mean_feat)

    def to_mot_format(self) -> List[str]:
        """Convert tracklet to MOT format strings.

        Format: frame_id, track_id, x, y, w, h, conf, class_id, -1, -1

        Returns:
            List of MOT format strings
        """
        lines = []
        for i, frame_id in enumerate(self.times):
            bbox = self.bboxes[i]
            score = self.scores[i] if i < len(self.scores) else 1.0
            line = f"{frame_id},{self.track_id},{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f},{score:.4f},{self.class_id},-1,-1"
            lines.append(line)
        return lines

    def __repr__(self):
        return f"Tracklet(id={self.track_id}, frames={self.start_frame}-{self.end_frame}, len={self.length})"

    def __len__(self):
        return self.length
