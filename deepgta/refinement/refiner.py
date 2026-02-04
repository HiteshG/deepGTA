"""
GTA-Link Tracklet Refinement for DeepGTA

Provides offline tracklet refinement through:
- DBSCAN-based identity switch detection and splitting
- Feature-based tracklet merging with spatial constraints
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from .tracklet import Tracklet


class GTALinkRefiner:
    """GTA-Link offline tracklet refinement.

    Performs identity switch detection and correction through:
    1. Splitting: Detect ID switches using DBSCAN clustering on ReID features
    2. Merging: Connect tracklets that belong to the same identity
    """

    def __init__(self, config):
        """Initialize the refiner.

        Args:
            config: DeepGTAConfig instance
        """
        self.config = config

        # Splitting parameters
        self.split_eps = config.split_eps
        self.split_min_samples = config.split_min_samples
        self.split_max_k = config.split_max_k
        self.split_min_len = config.split_min_len

        # Merging parameters
        self.merge_dist_thresh = config.merge_dist_thresh
        self.spatial_factor = config.spatial_factor

        self.verbose = config.verbose

    def refine(self, tracks: List) -> Dict[int, Tracklet]:
        """Refine tracking results.

        Args:
            tracks: List of STrack objects or tracking results

        Returns:
            Dictionary mapping track_id to refined Tracklet
        """
        # Convert tracks to tracklets
        tracklets = self._convert_to_tracklets(tracks)

        if not tracklets:
            return {}

        # Get spatial constraints
        max_x_range, max_y_range = self._get_spatial_constraints(tracklets)

        # Step 1: Split tracklets with ID switches
        if self.verbose:
            print(f"Number of tracklets before splitting: {len(tracklets)}")

        split_tracklets = self._split_tracklets(tracklets)

        if self.verbose:
            print(f"Number of tracklets after splitting: {len(split_tracklets)}")

        # Step 2: Compute distance matrix
        dist_matrix = self._get_distance_matrix(split_tracklets)

        # Step 3: Merge tracklets
        merged_tracklets = self._merge_tracklets(
            split_tracklets, dist_matrix,
            max_x_range, max_y_range
        )

        if self.verbose:
            print(f"Number of tracklets after merging: {len(merged_tracklets)}")

        return merged_tracklets

    def _convert_to_tracklets(self, tracks: List) -> Dict[int, Tracklet]:
        """Convert track list/snapshots to Tracklet dictionary.

        Args:
            tracks: List of track objects or snapshot dictionaries

        Returns:
            Dictionary mapping track_id to Tracklet
        """
        tracklets = {}

        for track in tracks:
            # Handle dictionary snapshots (new format)
            if isinstance(track, dict):
                tid = track['track_id']
                frame_id = track.get('frame_id', 0)
                score = track.get('score', 1.0)
                tlwh = track.get('tlwh', [0, 0, 0, 0])
                class_id = track.get('class_id', 0)
                smooth_feat = track.get('smooth_feat')

                bbox = list(tlwh)

                if tid not in tracklets:
                    tracklets[tid] = Tracklet(
                        track_id=tid,
                        frames=frame_id,
                        scores=score,
                        bboxes=bbox,
                        class_id=class_id
                    )
                else:
                    tracklets[tid].append_det(frame_id, score, bbox)

                # Add feature if available
                if smooth_feat is not None:
                    feat_copy = smooth_feat.copy() if hasattr(smooth_feat, 'copy') else smooth_feat
                    tracklets[tid].append_feat(feat_copy)

            # Handle STrack objects (legacy format)
            elif hasattr(track, 'track_id'):
                tid = track.track_id
                frame_id = track.frame_id if hasattr(track, 'frame_id') else 0
                score = track.score if hasattr(track, 'score') else 1.0
                tlwh = track.tlwh if hasattr(track, 'tlwh') else track._tlwh
                class_id = track.class_id if hasattr(track, 'class_id') else 0

                bbox = list(tlwh)

                if tid not in tracklets:
                    tracklets[tid] = Tracklet(
                        track_id=tid,
                        frames=frame_id,
                        scores=score,
                        bboxes=bbox,
                        class_id=class_id
                    )
                else:
                    tracklets[tid].append_det(frame_id, score, bbox)

                # Add feature if available
                if hasattr(track, 'smooth_feat') and track.smooth_feat is not None:
                    tracklets[tid].append_feat(track.smooth_feat.copy())
                elif hasattr(track, 'curr_feat') and track.curr_feat is not None:
                    tracklets[tid].append_feat(track.curr_feat.copy())

        if self.verbose:
            print(f"Converted {len(tracks)} snapshots to {len(tracklets)} tracklets")

        return tracklets

    def _get_spatial_constraints(self, tracklets: Dict[int, Tracklet]) -> Tuple[float, float]:
        """Calculate spatial constraints for merging.

        Args:
            tracklets: Dictionary of tracklets

        Returns:
            Tuple of (max_x_range, max_y_range)
        """
        min_x, max_x = float('inf'), -float('inf')
        min_y, max_y = float('inf'), -float('inf')

        for track in tracklets.values():
            for bbox in track.bboxes:
                x, y, w, h = bbox[:4]
                cx = x + w / 2
                cy = y + h / 2
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

        x_range = abs(max_x - min_x) * self.spatial_factor
        y_range = abs(max_y - min_y) * self.spatial_factor

        return x_range, y_range

    def _split_tracklets(self, tracklets: Dict[int, Tracklet]) -> Dict[int, Tracklet]:
        """Split tracklets with detected ID switches.

        Args:
            tracklets: Input tracklets

        Returns:
            Tracklets after splitting
        """
        new_id = max(tracklets.keys()) + 1
        result = {}

        iterator = tqdm(sorted(tracklets.keys()), desc="Splitting tracklets") if self.verbose else sorted(tracklets.keys())

        for tid in iterator:
            trklet = tracklets[tid]

            # Skip short tracklets
            if len(trklet.times) < self.split_min_len:
                result[tid] = trklet
                continue

            # Skip if no features
            if not trklet.features:
                result[tid] = trklet
                continue

            # Detect ID switch using clustering
            id_switch, labels = self._detect_id_switch(trklet.features)

            if not id_switch:
                result[tid] = trklet
            else:
                # Split into multiple tracklets
                embs = np.stack(trklet.features)
                frames = np.array(trklet.times)
                bboxes = np.stack(trklet.bboxes)
                scores = np.array(trklet.scores)

                unique_labels = set(labels)

                for label in unique_labels:
                    if label == -1:
                        continue  # Skip noise

                    mask = labels == label
                    tmp_embs = embs[mask]
                    tmp_frames = frames[mask]
                    tmp_bboxes = bboxes[mask]
                    tmp_scores = scores[mask]

                    new_tracklet = Tracklet(
                        track_id=new_id,
                        frames=tmp_frames.tolist(),
                        scores=tmp_scores.tolist(),
                        bboxes=tmp_bboxes.tolist(),
                        feats=tmp_embs.tolist(),
                        class_id=trklet.class_id
                    )
                    new_tracklet.parent_id = tid
                    result[new_id] = new_tracklet
                    new_id += 1

        return result

    def _detect_id_switch(self, features: List[np.ndarray]) -> Tuple[bool, np.ndarray]:
        """Detect ID switch using DBSCAN clustering.

        Args:
            features: List of feature vectors

        Returns:
            Tuple of (id_switch_detected, cluster_labels)
        """
        if len(features) > 15000:
            features = features[1::2]

        embs = np.stack(features)

        # Standardize
        scaler = StandardScaler()
        embs_scaled = scaler.fit_transform(embs)

        # DBSCAN clustering
        db = DBSCAN(
            eps=self.split_eps,
            min_samples=self.split_min_samples,
            metric='cosine'
        ).fit(embs_scaled)

        labels = db.labels_

        # Count clusters (excluding noise)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]

        # Assign noise to nearest cluster
        if -1 in labels and len(unique_labels) > 1:
            cluster_centers = np.array([
                embs_scaled[labels == label].mean(axis=0)
                for label in unique_labels
            ])

            noise_indices = np.where(labels == -1)[0]
            for idx in noise_indices:
                distances = cdist([embs_scaled[idx]], cluster_centers, metric='cosine')
                nearest_cluster = np.argmin(distances)
                labels[idx] = list(unique_labels)[nearest_cluster]

        n_clusters = len(unique_labels)

        # Merge clusters if too many
        if self.split_max_k and n_clusters > self.split_max_k:
            while n_clusters > self.split_max_k:
                cluster_centers = np.array([
                    embs_scaled[labels == label].mean(axis=0)
                    for label in unique_labels
                ])
                distance_matrix = cdist(cluster_centers, cluster_centers, metric='cosine')
                np.fill_diagonal(distance_matrix, np.inf)

                min_dist_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
                cluster_to_merge_1 = unique_labels[min_dist_idx[0]]
                cluster_to_merge_2 = unique_labels[min_dist_idx[1]]

                labels[labels == cluster_to_merge_2] = cluster_to_merge_1
                unique_labels = np.unique(labels)
                unique_labels = unique_labels[unique_labels != -1]
                n_clusters = len(unique_labels)

        return n_clusters > 1, labels

    def _get_distance_matrix(self, tracklets: Dict[int, Tracklet]) -> np.ndarray:
        """Compute distance matrix between tracklets.

        Args:
            tracklets: Dictionary of tracklets

        Returns:
            Distance matrix
        """
        n = len(tracklets)
        dist = np.zeros((n, n))
        tid_list = list(tracklets.keys())

        for i, tid1 in enumerate(tid_list):
            track1 = tracklets[tid1]
            for j, tid2 in enumerate(tid_list):
                if j < i:
                    dist[i][j] = dist[j][i]
                else:
                    track2 = tracklets[tid2]
                    dist[i][j] = self._get_distance(track1, track2)

        return dist

    def _get_distance(self, track1: Tracklet, track2: Tracklet) -> float:
        """Compute distance between two tracklets.

        Args:
            track1: First tracklet
            track2: Second tracklet

        Returns:
            Cosine distance (0-1)
        """
        # Check for temporal overlap
        if track1.track_id != track2.track_id:
            overlap = set(track1.times) & set(track2.times)
            if overlap:
                return 1.0  # Max distance for overlapping tracks

        # Compute cosine distance from features
        if not track1.features or not track2.features:
            return 1.0

        feat1 = np.stack(track1.features)
        feat2 = np.stack(track2.features)

        # Pairwise cosine similarity
        cos_sim = np.dot(feat1, feat2.T)
        norm1 = np.linalg.norm(feat1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(feat2, axis=1, keepdims=True)
        cos_sim = cos_sim / (norm1 @ norm2.T + 1e-8)

        cos_dist = 1 - cos_sim
        avg_dist = cos_dist.mean()

        return float(avg_dist)

    def _merge_tracklets(
        self,
        tracklets: Dict[int, Tracklet],
        dist_matrix: np.ndarray,
        max_x_range: float,
        max_y_range: float
    ) -> Dict[int, Tracklet]:
        """Merge tracklets based on feature similarity and spatial constraints.

        Args:
            tracklets: Dictionary of tracklets
            dist_matrix: Distance matrix
            max_x_range: Maximum x distance for merging
            max_y_range: Maximum y distance for merging

        Returns:
            Merged tracklets
        """
        tid_list = list(tracklets.keys())
        idx2tid = {idx: tid for idx, tid in enumerate(tid_list)}

        # Hierarchical merging
        diagonal_mask = np.eye(dist_matrix.shape[0], dtype=bool)
        non_diagonal_mask = ~diagonal_mask

        while np.any(dist_matrix[non_diagonal_mask] < self.merge_dist_thresh):
            # Find minimum distance pair
            min_index = np.argmin(dist_matrix[non_diagonal_mask])
            min_value = np.min(dist_matrix[non_diagonal_mask])

            masked_indices = np.where(non_diagonal_mask)
            track1_idx = masked_indices[0][min_index]
            track2_idx = masked_indices[1][min_index]

            track1 = tracklets[idx2tid[track1_idx]]
            track2 = tracklets[idx2tid[track2_idx]]

            # Check spatial constraints
            in_range = self._check_spatial_constraints(
                track1, track2, max_x_range, max_y_range
            )

            if in_range:
                # Merge track2 into track1
                track1.times.extend(track2.times)
                track1.scores.extend(track2.scores)
                track1.bboxes.extend(track2.bboxes)
                track1.features.extend(track2.features)

                # Update dictionary
                tracklets[idx2tid[track1_idx]] = track1
                del tracklets[idx2tid[track2_idx]]

                # Update distance matrix
                dist_matrix = np.delete(dist_matrix, track2_idx, axis=0)
                dist_matrix = np.delete(dist_matrix, track2_idx, axis=1)

                # Update index mapping
                idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}

                # Update distances for merged tracklet
                for idx in range(dist_matrix.shape[0]):
                    dist_matrix[track1_idx, idx] = self._get_distance(
                        tracklets[idx2tid[track1_idx]],
                        tracklets[idx2tid[idx]]
                    )
                    dist_matrix[idx, track1_idx] = dist_matrix[track1_idx, idx]

                # Update masks
                diagonal_mask = np.eye(dist_matrix.shape[0], dtype=bool)
                non_diagonal_mask = ~diagonal_mask
            else:
                # Mark as not mergeable
                dist_matrix[track1_idx, track2_idx] = self.merge_dist_thresh
                dist_matrix[track2_idx, track1_idx] = self.merge_dist_thresh

        return tracklets

    def _check_spatial_constraints(
        self,
        track1: Tracklet,
        track2: Tracklet,
        max_x_range: float,
        max_y_range: float
    ) -> bool:
        """Check if two tracklets satisfy spatial constraints for merging.

        Args:
            track1: First tracklet
            track2: Second tracklet
            max_x_range: Maximum allowed x distance
            max_y_range: Maximum allowed y distance

        Returns:
            True if constraints are satisfied
        """
        # Get consecutive segments
        seg1 = self._find_consecutive_segments(track1.times)
        seg2 = self._find_consecutive_segments(track2.times)

        # Get all subtracks sorted by time
        subtracks = self._get_sorted_subtracks(seg1, seg2, track1, track2)

        if len(subtracks) <= 1:
            return True

        # Check consecutive pairs
        prev = subtracks[0]
        for curr in subtracks[1:]:
            if prev.parent_id == curr.parent_id:
                prev = curr
                continue

            # Get exit location of prev and entry location of curr
            prev_bbox = prev.bboxes[-1]
            curr_bbox = curr.bboxes[0]

            x1 = prev_bbox[0] + prev_bbox[2] / 2
            y1 = prev_bbox[1] + prev_bbox[3] / 2
            x2 = curr_bbox[0] + curr_bbox[2] / 2
            y2 = curr_bbox[1] + curr_bbox[3] / 2

            dx = abs(x1 - x2)
            dy = abs(y1 - y2)

            if dx > max_x_range or dy > max_y_range:
                return False

            prev = curr

        return True

    @staticmethod
    def _find_consecutive_segments(times: List[int]) -> List[Tuple[int, int]]:
        """Find consecutive frame segments.

        Args:
            times: List of frame numbers

        Returns:
            List of (start_idx, end_idx) tuples
        """
        if not times:
            return []

        segments = []
        start_idx = 0
        end_idx = 0

        for i in range(1, len(times)):
            if times[i] == times[end_idx] + 1:
                end_idx = i
            else:
                segments.append((start_idx, end_idx))
                start_idx = i
                end_idx = i

        segments.append((start_idx, end_idx))
        return segments

    def _get_sorted_subtracks(
        self,
        seg1: List[Tuple[int, int]],
        seg2: List[Tuple[int, int]],
        track1: Tracklet,
        track2: Tracklet
    ) -> List[Tracklet]:
        """Get subtracks from both tracks sorted by start frame.

        Args:
            seg1: Segments from track1
            seg2: Segments from track2
            track1: First tracklet
            track2: Second tracklet

        Returns:
            Sorted list of subtracks
        """
        subtracks = []

        for start_idx, end_idx in seg1:
            subtrack = track1.extract(start_idx, end_idx)
            subtrack.parent_id = track1.track_id
            subtracks.append(subtrack)

        for start_idx, end_idx in seg2:
            subtrack = track2.extract(start_idx, end_idx)
            subtrack.parent_id = track2.track_id
            subtracks.append(subtrack)

        # Sort by start frame
        subtracks.sort(key=lambda t: t.start_frame)

        return subtracks
