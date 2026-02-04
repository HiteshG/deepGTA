"""
Base Track Class for DeepGTA

Provides the base class and state enumeration for tracks.
"""

import numpy as np
from collections import OrderedDict


class TrackState:
    """Enumeration of possible track states."""
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4


class BaseTrack:
    """Base class for all tracks.

    Provides common attributes and methods for track management.
    """

    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # Multi-camera support
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        """Return the last frame where this track was active."""
        return self.frame_id

    @staticmethod
    def next_id():
        """Generate next unique track ID."""
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        """Activate the track."""
        raise NotImplementedError

    def predict(self):
        """Predict the next state."""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """Update the track with new observation."""
        raise NotImplementedError

    def mark_lost(self):
        """Mark the track as lost."""
        self.state = TrackState.Lost

    def mark_long_lost(self):
        """Mark the track as long lost."""
        self.state = TrackState.LongLost

    def mark_removed(self):
        """Mark the track as removed."""
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        """Reset the track ID counter."""
        BaseTrack._count = 0
