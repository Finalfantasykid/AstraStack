"""Tests for Video.py."""
from unittest import mock

import numpy as np

from FrameSource import ImageSequenceSource
from tests.inputs import (
    PNG8_MIXED_SIZE_SEQUENCE,
    PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
    PNG8_SEQUENCE,
    PNG8_SEQUENCE_DIMENSIONS,
)
from Video import loadFrames


def test_loadFrames_leaves_out_unloadable_frames() -> None:
    """Frames of another size, and indices past the last frame, are left out."""
    frames, sharps, deltas = loadFrames(
        source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
        indices=range(0, 6),
        frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
        progress=mock.Mock(),
    )

    assert frames == [0, 2, 3]
    assert len(sharps) == len(deltas) == len(frames)


def test_loadFrames_uses_frame_before_chunk() -> None:
    """A chunk's first delta is taken from the frame before the chunk, as it is when loading all frames."""
    allFrames = loadFrames(
        source=ImageSequenceSource(PNG8_SEQUENCE),
        indices=range(0, 4),
        frameSize=PNG8_SEQUENCE_DIMENSIONS,
        progress=mock.Mock(),
    )

    frames, sharps, deltas = loadFrames(
        source=ImageSequenceSource(PNG8_SEQUENCE),
        indices=range(3, 4),
        frameSize=PNG8_SEQUENCE_DIMENSIONS,
        progress=mock.Mock(),
    )

    assert frames == [3]
    assert sharps == allFrames[1][-1:]
    assert np.array_equal(deltas[0], allFrames[2][-1])
