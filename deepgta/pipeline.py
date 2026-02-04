"""
DeepGTA Pipeline

Main unified pipeline that combines detection, tracking, and refinement
for end-to-end multi-object tracking.
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .config import DeepGTAConfig
from .detection import YOLODetector
from .tracking import DeepEIoUTracker
from .refinement import GTALinkRefiner, Tracklet
from .reid import ReIDExtractor
from .utils import VideoReader, VideoWriter, Visualizer


class DeepGTAPipeline:
    """End-to-end multi-object tracking pipeline.

    Combines:
    1. YOLOv11x detection with configurable class handling
    2. Deep-EIoU online tracking with ReID features
    3. GTA-Link offline refinement (optional)
    """

    def __init__(self, config: DeepGTAConfig):
        """Initialize the pipeline.

        Args:
            config: DeepGTAConfig instance with all settings
        """
        self.config = config

        # Initialize components
        self.detector = YOLODetector(config)
        self.tracker = DeepEIoUTracker(config, frame_rate=config.frame_rate)

        # Initialize ReID extractor if enabled
        if config.with_reid and config.reid_model:
            self.reid_extractor = ReIDExtractor(
                model_path=config.reid_model,
                device=config.device
            )
        else:
            self.reid_extractor = None

        # Initialize refiner if enabled
        if config.use_refinement:
            self.refiner = GTALinkRefiner(config)
        else:
            self.refiner = None

        # Visualizer
        self.visualizer = Visualizer(
            class_colors=config.track_colors,
            thickness=config.track_thickness,
            font_scale=config.label_font_scale
        )

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_progress: bool = True
    ) -> Tuple[str, Dict[int, Tracklet]]:
        """Process a video through the full pipeline.

        Args:
            video_path: Path to input video
            output_path: Path for output video (optional)
            show_progress: Show progress bar

        Returns:
            Tuple of (output_video_path, tracklets_dict)
        """
        # Stage 1: Online detection and tracking
        all_tracks, frame_tracks = self._online_tracking(video_path, show_progress)

        # Stage 2: Offline refinement (if enabled)
        if self.refiner is not None:
            if self.config.verbose:
                print("Running GTA-Link refinement...")
            tracklets = self.refiner.refine(all_tracks)
        else:
            tracklets = self._tracks_to_tracklets(all_tracks)

        # Stage 3: Render output video (if requested)
        if self.config.output_video and output_path:
            self._render_video(video_path, output_path, tracklets, show_progress)
        elif output_path is None and self.config.output_video:
            # Generate default output path
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_tracked{ext}"
            self._render_video(video_path, output_path, tracklets, show_progress)

        # Save MOT format results (if enabled)
        if self.config.save_mot_txt:
            mot_path = self.config.mot_output_path
            if mot_path is None:
                base, _ = os.path.splitext(video_path)
                mot_path = f"{base}_results.txt"
            self._save_mot_results(tracklets, mot_path)

        return output_path or "", tracklets

    def _online_tracking(
        self,
        video_path: str,
        show_progress: bool = True
    ) -> Tuple[List, Dict[int, List]]:
        """Run online detection and tracking.

        Args:
            video_path: Path to input video
            show_progress: Show progress bar

        Returns:
            Tuple of (all_track_snapshots, frame_tracks_dict)
        """
        # Store snapshots of track state (not references to mutable objects!)
        all_track_snapshots = []
        frame_tracks = {}

        with VideoReader(video_path) as reader:
            # Update tracker frame rate
            self.tracker.frame_rate = reader.fps

            iterator = tqdm(reader, total=len(reader), desc="Tracking") if show_progress else reader

            for frame_id, frame in iterator:
                # Detection
                if self.config.verbose and frame_id <= 3:
                    print(f"\nFrame {frame_id}:")

                detections = self.detector.detect(frame)

                if self.config.verbose and frame_id <= 3:
                    print(f"  Filtered detections: {len(detections)}")

                # Extract ReID features
                if self.reid_extractor is not None and detections:
                    embeddings = self.reid_extractor.extract_from_frame(frame, detections)
                else:
                    embeddings = None

                # Update tracker
                tracks = self.tracker.update(detections, embeddings)

                # Store snapshots of track state at this frame
                # IMPORTANT: We must copy the data, not store references!
                frame_tracks[frame_id] = []
                for track in tracks:
                    # Create a snapshot dictionary with current state
                    snapshot = {
                        'track_id': track.track_id,
                        'frame_id': frame_id,  # Use current frame_id, not track.frame_id
                        'score': track.score,
                        'tlwh': track.tlwh.copy(),  # Copy the array!
                        'class_id': getattr(track, 'class_id', 0),
                        'smooth_feat': track.smooth_feat.copy() if track.smooth_feat is not None else None
                    }
                    all_track_snapshots.append(snapshot)
                    frame_tracks[frame_id].append(snapshot)

                if self.config.verbose and frame_id <= 3:
                    print(f"  Active tracks: {len(tracks)}")
                elif self.config.verbose and frame_id % 100 == 0:
                    print(f"Frame {frame_id}: {len(detections)} detections, {len(tracks)} active tracks")

        if self.config.verbose:
            print(f"\nTotal track snapshots collected: {len(all_track_snapshots)}")

        return all_track_snapshots, frame_tracks

    def _tracks_to_tracklets(self, track_snapshots: List) -> Dict[int, Tracklet]:
        """Convert track snapshots to tracklet dictionary.

        Args:
            track_snapshots: List of track snapshot dictionaries

        Returns:
            Dictionary mapping track_id to Tracklet
        """
        tracklets = {}

        for snapshot in track_snapshots:
            # Handle both dict snapshots and STrack objects for compatibility
            if isinstance(snapshot, dict):
                tid = snapshot['track_id']
                frame_id = snapshot['frame_id']
                score = snapshot['score']
                tlwh = snapshot['tlwh']
                class_id = snapshot.get('class_id', 0)
                smooth_feat = snapshot.get('smooth_feat')
            else:
                # Legacy STrack object support
                tid = snapshot.track_id
                frame_id = snapshot.frame_id
                score = snapshot.score
                tlwh = snapshot.tlwh
                class_id = getattr(snapshot, 'class_id', 0)
                smooth_feat = getattr(snapshot, 'smooth_feat', None)

            if tid not in tracklets:
                tracklets[tid] = Tracklet(
                    track_id=tid,
                    frames=frame_id,
                    scores=score,
                    bboxes=list(tlwh),
                    class_id=class_id
                )
            else:
                tracklets[tid].append_det(frame_id, score, list(tlwh))

            # Add features if available
            if smooth_feat is not None:
                tracklets[tid].append_feat(smooth_feat.copy() if hasattr(smooth_feat, 'copy') else smooth_feat)

        return tracklets

    def _render_video(
        self,
        video_path: str,
        output_path: str,
        tracklets: Dict[int, Tracklet],
        show_progress: bool = True
    ):
        """Render output video with tracking results.

        Args:
            video_path: Input video path
            output_path: Output video path
            tracklets: Tracking results
            show_progress: Show progress bar
        """
        # Build frame-to-tracks lookup
        frame_results = {}
        total_entries = 0

        for tid, tracklet in tracklets.items():
            if self.config.verbose:
                print(f"Tracklet {tid}: {len(tracklet.times)} frames, "
                      f"range [{min(tracklet.times) if tracklet.times else 0}-{max(tracklet.times) if tracklet.times else 0}]")

            for i, frame_id in enumerate(tracklet.times):
                # Ensure frame_id is an integer
                frame_id = int(frame_id)

                if frame_id not in frame_results:
                    frame_results[frame_id] = []

                # Ensure bbox is a list
                bbox = tracklet.bboxes[i]
                if hasattr(bbox, 'tolist'):
                    bbox = bbox.tolist()

                frame_results[frame_id].append({
                    'track_id': tid,
                    'bbox': bbox,
                    'score': tracklet.scores[i] if i < len(tracklet.scores) else 1.0,
                    'class_id': tracklet.class_id
                })
                total_entries += 1

        if self.config.verbose:
            print(f"\nTotal tracklets: {len(tracklets)}")
            print(f"Total frame entries: {total_entries}")
            print(f"Frames with tracks: {len(frame_results)}")
            if frame_results:
                print(f"Frame range: {min(frame_results.keys())} - {max(frame_results.keys())}")

        with VideoReader(video_path) as reader:
            with VideoWriter.from_reader(reader, output_path, self.config.output_codec) as writer:
                iterator = tqdm(reader, total=len(reader), desc="Rendering") if show_progress else reader

                for frame_id, frame in iterator:
                    # Get tracks for this frame
                    tracks = frame_results.get(frame_id, [])

                    # Draw tracks
                    if tracks:
                        frame = self._draw_frame_tracks(frame, tracks)

                    # Draw frame info
                    if self.config.draw_labels:
                        frame = self.visualizer.draw_frame_info(
                            frame, frame_id, len(tracks)
                        )

                    writer.write(frame)

        if self.config.verbose:
            print(f"Output video saved to: {output_path}")

    def _draw_frame_tracks(self, frame: np.ndarray, tracks: List[dict]) -> np.ndarray:
        """Draw tracks on a frame.

        Args:
            frame: Input frame
            tracks: List of track dictionaries

        Returns:
            Frame with drawn tracks
        """
        output = frame.copy()

        for track in tracks:
            bbox = track['bbox']
            track_id = track['track_id']
            class_id = track.get('class_id', 0)
            score = track.get('score', 1.0)

            # Convert tlwh to tlbr
            x, y, w, h = bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

            # Get color
            from .utils.visualization import get_color_for_id
            color = get_color_for_id(track_id)

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.config.track_thickness)

            # Draw label
            if self.config.draw_labels:
                if class_id < len(self.config.class_names):
                    class_name = self.config.class_names[class_id]
                else:
                    class_name = f"cls{class_id}"

                label = f"ID:{track_id} {class_name}"

                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.config.label_font_scale, 1
                )

                cv2.rectangle(
                    output,
                    (x1, y1 - text_h - 5),
                    (x1 + text_w + 2, y1),
                    color,
                    -1
                )

                cv2.putText(
                    output,
                    label,
                    (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.label_font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        return output

    def _save_mot_results(self, tracklets: Dict[int, Tracklet], output_path: str):
        """Save tracking results in MOT format.

        Args:
            tracklets: Tracking results
            output_path: Output file path
        """
        results = []

        # Reassign track IDs sequentially
        for new_id, (tid, tracklet) in enumerate(sorted(tracklets.items()), 1):
            for i, frame_id in enumerate(tracklet.times):
                bbox = tracklet.bboxes[i]
                score = tracklet.scores[i] if i < len(tracklet.scores) else 1.0
                results.append([
                    frame_id, new_id,
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    score, tracklet.class_id, -1, -1
                ])

        # Sort by frame
        results.sort(key=lambda x: (x[0], x[1]))

        # Write to file
        with open(output_path, 'w') as f:
            for line in results:
                f.write(f"{line[0]},{line[1]},{line[2]:.2f},{line[3]:.2f},"
                        f"{line[4]:.2f},{line[5]:.2f},{line[6]:.4f},"
                        f"{line[7]},{line[8]},{line[9]}\n")

        if self.config.verbose:
            print(f"MOT results saved to: {output_path}")

    def reset(self):
        """Reset the pipeline state."""
        self.tracker.reset()

    def process_video_debug(
        self,
        video_path: str,
        output_path: str,
        skip_refinement: bool = True
    ) -> Tuple[str, Dict[int, Tracklet]]:
        """Process video with debug options.

        Args:
            video_path: Path to input video
            output_path: Path for output video
            skip_refinement: If True, skip GTA-Link refinement

        Returns:
            Tuple of (output_video_path, tracklets_dict)
        """
        # Force verbose mode
        original_verbose = self.config.verbose
        self.config.verbose = True

        # Run online tracking
        all_track_snapshots, frame_tracks = self._online_tracking(video_path, show_progress=True)

        print(f"\n{'='*50}")
        print("DEBUG: Online tracking results")
        print('='*50)
        print(f"Total snapshots: {len(all_track_snapshots)}")
        print(f"Frames with tracks: {len(frame_tracks)}")

        # Sample some frames
        sample_frames = sorted(frame_tracks.keys())[:5]
        for fid in sample_frames:
            tracks = frame_tracks[fid]
            print(f"  Frame {fid}: {len(tracks)} tracks")

        # Convert to tracklets
        if skip_refinement:
            print("\nSkipping GTA-Link refinement (debug mode)")
            tracklets = self._tracks_to_tracklets(all_track_snapshots)
        else:
            print("\nRunning GTA-Link refinement")
            tracklets = self.refiner.refine(all_track_snapshots)

        print(f"\n{'='*50}")
        print("DEBUG: Tracklet results")
        print('='*50)
        print(f"Total tracklets: {len(tracklets)}")

        for tid, t in list(tracklets.items())[:5]:
            print(f"  Tracklet {tid}: frames={len(t.times)}, "
                  f"bboxes={len(t.bboxes)}, "
                  f"range=[{min(t.times) if t.times else 0}-{max(t.times) if t.times else 0}]")

        # Render
        print(f"\n{'='*50}")
        print("DEBUG: Rendering")
        print('='*50)
        self._render_video(video_path, output_path, tracklets, show_progress=True)

        # Restore verbose setting
        self.config.verbose = original_verbose

        return output_path, tracklets


# Import cv2 at module level for _draw_frame_tracks
import cv2
