"""
Video I/O Utilities for DeepGTA

Provides video reading and writing functionality with
frame iteration and metadata handling.
"""

import cv2
import numpy as np
from typing import Iterator, Optional, Tuple


class VideoReader:
    """Video reader with frame iteration support.

    Provides an iterator interface for reading video frames
    along with video metadata.
    """

    def __init__(self, video_path: str):
        """Initialize the video reader.

        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame = 0

    @property
    def width(self) -> int:
        """Video frame width."""
        return self._width

    @property
    def height(self) -> int:
        """Video frame height."""
        return self._height

    @property
    def fps(self) -> float:
        """Video frame rate."""
        return self._fps

    @property
    def frame_count(self) -> int:
        """Total number of frames."""
        return self._frame_count

    @property
    def resolution(self) -> Tuple[int, int]:
        """Video resolution (width, height)."""
        return (self._width, self._height)

    @property
    def current_frame(self) -> int:
        """Current frame index (0-based)."""
        return self._current_frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame.

        Returns:
            Tuple of (success, frame) where frame is BGR numpy array
        """
        ret, frame = self.cap.read()
        if ret:
            self._current_frame += 1
        return ret, frame

    def seek(self, frame_idx: int):
        """Seek to a specific frame.

        Args:
            frame_idx: Frame index to seek to (0-based)
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._current_frame = frame_idx

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Iterate over video frames.

        Yields:
            Tuple of (frame_id, frame) where frame_id is 1-indexed
        """
        self.seek(0)
        while True:
            ret, frame = self.read()
            if not ret:
                break
            yield self._current_frame, frame

    def __len__(self) -> int:
        """Total number of frames."""
        return self._frame_count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self):
        """Release the video capture."""
        if self.cap is not None:
            self.cap.release()


class VideoWriter:
    """Video writer with frame-by-frame writing support."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float = 30.0,
        codec: str = 'mp4v'
    ):
        """Initialize the video writer.

        Args:
            output_path: Output video file path
            width: Frame width
            height: Frame height
            fps: Frame rate
            codec: FourCC codec string
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not self.writer.isOpened():
            raise ValueError(f"Could not create video writer: {output_path}")

        self._frame_count = 0

    def write(self, frame: np.ndarray):
        """Write a frame to the video.

        Args:
            frame: BGR numpy array
        """
        # Ensure correct size
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        self.writer.write(frame)
        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        """Number of frames written."""
        return self._frame_count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self):
        """Release the video writer."""
        if self.writer is not None:
            self.writer.release()

    @classmethod
    def from_reader(
        cls,
        reader: VideoReader,
        output_path: str,
        codec: str = 'mp4v'
    ) -> 'VideoWriter':
        """Create a writer with the same properties as a reader.

        Args:
            reader: VideoReader instance
            output_path: Output video file path
            codec: FourCC codec string

        Returns:
            VideoWriter instance
        """
        return cls(
            output_path=output_path,
            width=reader.width,
            height=reader.height,
            fps=reader.fps,
            codec=codec
        )


def extract_frames(video_path: str, output_dir: str, ext: str = 'jpg') -> int:
    """Extract all frames from a video to a directory.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        ext: Frame file extension

    Returns:
        Number of frames extracted
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    with VideoReader(video_path) as reader:
        for frame_id, frame in reader:
            frame_path = os.path.join(output_dir, f"{frame_id:06d}.{ext}")
            cv2.imwrite(frame_path, frame)

    return reader.frame_count


def get_video_info(video_path: str) -> dict:
    """Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video properties
    """
    with VideoReader(video_path) as reader:
        return {
            'path': video_path,
            'width': reader.width,
            'height': reader.height,
            'fps': reader.fps,
            'frame_count': reader.frame_count,
            'duration': reader.frame_count / reader.fps if reader.fps > 0 else 0
        }
