"""
Matching Functions for DeepGTA

Provides distance metrics and linear assignment for track association.
Includes IoU, EIoU, and embedding distance calculations.
"""

import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from .kalman_filter import chi2inv95


def linear_assignment(cost_matrix, thresh):
    """Solve linear assignment problem.

    Args:
        cost_matrix: Cost matrix of shape (N, M)
        thresh: Maximum cost threshold

    Returns:
        Tuple of (matches, unmatched_a, unmatched_b) where matches is Kx2 array
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    matches = []
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """Compute IoU between two sets of boxes.

    Args:
        atlbrs: First set of boxes in [x1, y1, x2, y2] format
        btlbrs: Second set of boxes in [x1, y1, x2, y2] format

    Returns:
        IoU matrix of shape (len(atlbrs), len(btlbrs))
    """
    ious_matrix = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious_matrix.size == 0:
        return ious_matrix

    atlbrs = np.asarray(atlbrs, dtype=np.float64)
    btlbrs = np.asarray(btlbrs, dtype=np.float64)

    # Compute IoU manually
    for i, box_a in enumerate(atlbrs):
        for j, box_b in enumerate(btlbrs):
            # Intersection
            x1 = max(box_a[0], box_b[0])
            y1 = max(box_a[1], box_b[1])
            x2 = min(box_a[2], box_b[2])
            y2 = min(box_a[3], box_b[3])

            inter_w = max(0, x2 - x1)
            inter_h = max(0, y2 - y1)
            inter_area = inter_w * inter_h

            # Union
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            union_area = area_a + area_b - inter_area

            ious_matrix[i, j] = inter_area / (union_area + 1e-8)

    return ious_matrix


def expand_box(tlbr, e):
    """Expand bounding box by a factor.

    Args:
        tlbr: Box in [x1, y1, x2, y2] format
        e: Expansion factor

    Returns:
        Expanded box in [x1, y1, x2, y2] format
    """
    t, l, b, r = tlbr[1], tlbr[0], tlbr[3], tlbr[2]
    w = r - l
    h = b - t
    expand_w = 2 * w * e + w
    expand_h = 2 * h * e + h

    new_tlbr = [
        l - expand_w // 2,
        t - expand_h // 2,
        r + expand_w // 2,
        b + expand_h // 2
    ]

    return [new_tlbr[1], new_tlbr[0], new_tlbr[3], new_tlbr[2]]


def eious(atlbrs, btlbrs, e):
    """Compute expanded IoU (EIoU) between two sets of boxes.

    Args:
        atlbrs: First set of boxes in [x1, y1, x2, y2] format
        btlbrs: Second set of boxes in [x1, y1, x2, y2] format
        e: Expansion factor

    Returns:
        EIoU matrix of shape (len(atlbrs), len(btlbrs))
    """
    eious_matrix = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if eious_matrix.size == 0:
        return eious_matrix

    atlbrs_expanded = np.array([expand_box(tlbr, e) for tlbr in atlbrs])
    btlbrs_expanded = np.array([expand_box(tlbr, e) for tlbr in btlbrs])

    return ious(atlbrs_expanded, btlbrs_expanded)


def iou_distance(atracks, btracks):
    """Compute IoU distance between tracks.

    Args:
        atracks: List of tracks or array of boxes
        btracks: List of tracks or array of boxes

    Returns:
        Cost matrix (1 - IoU)
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
       (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix


def eiou_distance(atracks, btracks, expand):
    """Compute EIoU distance between tracks using last position.

    Args:
        atracks: List of tracks
        btracks: List of tracks (detections)
        expand: Expansion factor for EIoU

    Returns:
        Cost matrix (1 - EIoU)
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
       (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.last_tlbr for track in atracks]
        btlbrs = [track.last_tlbr for track in btracks]

    _ious = eious(atlbrs, btlbrs, expand)
    cost_matrix = 1 - _ious
    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """Compute embedding distance between tracks and detections.

    Args:
        tracks: List of tracks with smooth_feat attribute
        detections: List of detections with curr_feat attribute
        metric: Distance metric (default: cosine)

    Returns:
        Cost matrix of embedding distances
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))

    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """Gate cost matrix using Mahalanobis distance.

    Args:
        kf: Kalman filter instance
        cost_matrix: Cost matrix to gate
        tracks: List of tracks
        detections: List of detections
        only_position: Use only position for gating

    Returns:
        Gated cost matrix
    """
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xywh() for det in detections])

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf

    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    """Fuse appearance cost with motion cost.

    Args:
        kf: Kalman filter instance
        cost_matrix: Appearance cost matrix
        tracks: List of tracks
        detections: List of detections
        only_position: Use only position for motion
        lambda_: Weight for appearance vs motion

    Returns:
        Fused cost matrix
    """
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xywh() for det in detections])

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance

    return cost_matrix


def fuse_score(cost_matrix, detections):
    """Fuse IoU distance with detection scores.

    Args:
        cost_matrix: IoU distance matrix
        detections: List of detections with score attribute

    Returns:
        Score-fused cost matrix
    """
    if cost_matrix.size == 0:
        return cost_matrix

    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim

    return fuse_cost
