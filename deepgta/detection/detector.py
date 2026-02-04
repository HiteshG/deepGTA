"""
YOLO Detection Module for DeepGTA

Provides a configurable YOLO detector with support for:
- Multi-class filtering
- Special class handling (max-confidence only)
- Per-class NMS
- Multi-scale TTA (Test-Time Augmentation)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ultralytics import YOLO


@dataclass
class Detection:
    """Represents a single detection.

    Attributes:
        bbox: Bounding box in [x1, y1, x2, y2] format (tlbr)
        tlwh: Bounding box in [x, y, w, h] format
        confidence: Detection confidence score
        class_id: Class index
        class_name: Class name string
    """
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

    @property
    def tlwh(self) -> np.ndarray:
        """Get bounding box in [x, y, w, h] format."""
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1])

    @property
    def tlbr(self) -> np.ndarray:
        """Get bounding box in [x1, y1, x2, y2] format."""
        return self.bbox.copy()

    @property
    def xywh(self) -> np.ndarray:
        """Get bounding box in [cx, cy, w, h] format (center-based)."""
        x1, y1, x2, y2 = self.bbox
        w, h = x2 - x1, y2 - y1
        return np.array([x1 + w / 2, y1 + h / 2, w, h])


class YOLODetector:
    """YOLO-based object detector with configurable class handling.

    Supports filtering detections to specific classes and special handling
    for classes where only the maximum confidence detection should be kept.
    """

    def __init__(self, config):
        """Initialize the detector.

        Args:
            config: DeepGTAConfig instance with detection settings
        """
        self.config = config
        self.model = YOLO(config.yolo_weights)
        self.device = config.device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single frame.

        Args:
            frame: Input image as numpy array (BGR format)

        Returns:
            List of Detection objects for tracked classes
        """
        # Run inference
        if self.config.use_tta:
            detections = self._detect_with_tta(frame)
        else:
            detections = self._detect_single_scale(frame)

        # Filter to tracked classes
        detections = [d for d in detections if d.class_id in self.config.track_classes]

        # Handle special classes (keep max-confidence only)
        detections = self._handle_special_classes(detections)

        # Apply per-class NMS
        detections = self._per_class_nms(detections)

        return detections

    def _detect_single_scale(self, frame: np.ndarray) -> List[Detection]:
        """Run detection at a single scale."""
        imgsz = self.config.detection_imgsz[0]
        results = self.model(
            frame,
            imgsz=imgsz,
            conf=self.config.detection_conf_thresh,
            iou=self.config.detection_iou_thresh,
            device=self.device,
            verbose=False
        )

        return self._parse_results(results)

    def _detect_with_tta(self, frame: np.ndarray) -> List[Detection]:
        """Run detection with multi-scale TTA."""
        all_detections = []

        for imgsz in self.config.detection_imgsz:
            results = self.model(
                frame,
                imgsz=imgsz,
                conf=self.config.detection_conf_thresh,
                iou=self.config.detection_iou_thresh,
                device=self.device,
                verbose=False
            )
            all_detections.extend(self._parse_results(results))

        # Apply NMS to merge multi-scale detections
        if all_detections:
            all_detections = self._per_class_nms(all_detections)

        return all_detections

    def _parse_results(self, results) -> List[Detection]:
        """Parse YOLO results into Detection objects."""
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, classes):
                class_name = self.config.get_class_name(cls_id)
                det = Detection(
                    bbox=box.astype(np.float32),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=class_name
                )
                detections.append(det)

        return detections

    def _handle_special_classes(self, detections: List[Detection]) -> List[Detection]:
        """Keep only max-confidence detection for special classes.

        For classes like 'Puck' where there should only be one instance,
        keep only the detection with highest confidence.
        """
        if not self.config.special_classes:
            return detections

        # Separate special and regular detections
        regular_dets = []
        special_dets = {cls_id: [] for cls_id in self.config.special_classes}

        for det in detections:
            if det.class_id in self.config.special_classes:
                special_dets[det.class_id].append(det)
            else:
                regular_dets.append(det)

        # Keep only max-confidence detection for each special class
        for cls_id, dets in special_dets.items():
            if dets:
                max_det = max(dets, key=lambda d: d.confidence)
                regular_dets.append(max_det)

        return regular_dets

    def _per_class_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply NMS per class."""
        if not detections:
            return detections

        # Group by class
        class_dets = {}
        for det in detections:
            if det.class_id not in class_dets:
                class_dets[det.class_id] = []
            class_dets[det.class_id].append(det)

        # Apply NMS to each class
        nms_results = []
        for cls_id, dets in class_dets.items():
            if len(dets) <= 1:
                nms_results.extend(dets)
                continue

            boxes = np.array([d.bbox for d in dets])
            scores = np.array([d.confidence for d in dets])

            keep_indices = self._nms(boxes, scores, self.config.detection_iou_thresh)
            nms_results.extend([dets[i] for i in keep_indices])

        return nms_results

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
        """Non-maximum suppression implementation.

        Args:
            boxes: Array of boxes [N, 4] in xyxy format
            scores: Array of scores [N]
            iou_thresh: IoU threshold for suppression

        Returns:
            List of indices to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)

            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return keep

    def to_mot_format(self, frame_id: int, detections: List[Detection]) -> List[str]:
        """Convert detections to MOT Challenge format.

        Format: frame_id, -1, x, y, w, h, conf, class_id, -1, -1

        Args:
            frame_id: Current frame number (1-indexed)
            detections: List of Detection objects

        Returns:
            List of strings in MOT format
        """
        lines = []
        for det in detections:
            tlwh = det.tlwh
            line = f"{frame_id},-1,{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{det.confidence:.4f},{det.class_id},-1,-1"
            lines.append(line)
        return lines
