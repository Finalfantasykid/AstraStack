"""Tests for FrameSource.py."""
import pickle
import unittest
from collections.abc import Sequence

import av
import cv2
import ddt
import numpy as np

from FrameSource import FrameSource, ImageSequenceSource, VideoSource
from tests.inputs import (
    DNG16_RGGB_SEQUENCE,
    DNG16_RGGB_SEQUENCE_DIMENSIONS,
    DROPPED_FRAMES_AVI,
    FITS16_RGGB_SEQUENCE,
    FITS16_RGGB_SEQUENCE_DIMENSIONS,
    JPG8,
    JPG8_DIMENSIONS,
    MP4_YUV8,
    MP4_YUV8_DIMENSIONS,
    MP4_YUV8_FRAME_COUNT,
    MP4_YUV8_LONG_GOP,
    NOT_MEDIA,
    PNG16,
    PNG8_SEQUENCE,
    PNG8_SEQUENCE_DIMENSIONS,
)


@ddt.ddt
class TestImageSequenceSource(unittest.TestCase):
    """ImageSequenceSource."""

    def test_is_a_frameSource(self) -> None:
        """Image sequences are frame sources that keep their paths."""
        source = ImageSequenceSource(PNG8_SEQUENCE)

        assert source.paths == PNG8_SEQUENCE
        assert isinstance(source, FrameSource)

    def test_frameCount_and_size(self) -> None:
        """The count is the number of files and the size is the first file's (width, height)."""
        source = ImageSequenceSource(PNG8_SEQUENCE)
        height, width = cv2.imread(PNG8_SEQUENCE[0]).shape[:2]

        assert source.frameCount() == len(PNG8_SEQUENCE)
        assert source.frameSize() == (width, height)

    def test_readFrame_by_index(self) -> None:
        """Frames are read by index in any order, as stored."""
        source = ImageSequenceSource(PNG8_SEQUENCE)

        for index in [2, 0, 3, 1, 1, 2]:
            assert np.array_equal(source.readFrame(index), cv2.imread(PNG8_SEQUENCE[index], cv2.IMREAD_UNCHANGED))

    @ddt.data(
        # An 8-bit PNG reads as 8 bits.
        {"path": PNG8_SEQUENCE[0], "dtype": np.uint8},
        # A 16-bit PNG reads as 16 bits, not reduced to 8.
        {"path": PNG16, "dtype": np.uint16},
        # An 8-bit JPEG reads as 8 bits.
        {"path": JPG8, "dtype": np.uint8},
        # A raw camera file is decoded to 16 bits.
        {"path": DNG16_RGGB_SEQUENCE[0], "dtype": np.uint16},
    )
    @ddt.unpack
    def test_readFrame_color_native_depth(self, path: str, dtype: type) -> None:
        """Color frames are read with three channels at their stored depth, and frameSize agrees with them."""
        source = ImageSequenceSource([path])

        frame = source.readFrame(0)

        assert frame.dtype == dtype
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert source.frameSize() == (frame.shape[1], frame.shape[0])

    def test_readFrame_past_end(self) -> None:
        """There is no frame past the last file."""
        assert ImageSequenceSource(PNG8_SEQUENCE).readFrame(len(PNG8_SEQUENCE)) is None

    def test_pickles(self) -> None:
        """A source survives a pickle round trip and reads the same frames."""
        source = ImageSequenceSource(PNG8_SEQUENCE)
        before = source.readFrame(1)

        source = pickle.loads(pickle.dumps(source))

        assert source.paths == PNG8_SEQUENCE
        assert np.array_equal(source.readFrame(1), before)


@ddt.ddt
class TestVideoSource(unittest.TestCase):
    """VideoSource."""

    @staticmethod
    def _decodeFrames(path: str) -> list[np.ndarray]:
        """Reads every frame of a video in order with a fresh VideoSource, so it never seeks.

        Args:
            path: The video file.

        Returns:
            The frames, in order.
        """
        source = VideoSource(path)
        try:
            return [source.readFrame(index) for index in range(source.frameCount())]
        finally:
            source.close()

    def test_frameCount_and_size(self) -> None:
        """The count is the number of frames in the video and the size is the container's."""
        source = VideoSource(MP4_YUV8)

        assert source.frameCount() == MP4_YUV8_FRAME_COUNT
        assert source.frameSize() == MP4_YUV8_DIMENSIONS

    def test_frameCount_is_counted_once(self) -> None:
        """The frames are counted once, the first time the count is needed."""
        source = VideoSource(MP4_YUV8)
        assert source.timestamps is None
        # Reading just the first frame does not trigger a full read. This
        # behavior is what makes thumbnail generation quick.
        source.readFrame(0)
        assert source.timestamps is None

        assert source.frameCount() == MP4_YUV8_FRAME_COUNT
        source.path = "missing.avi"
        assert source.frameCount() == MP4_YUV8_FRAME_COUNT

    def test_readFrame_in_any_order(self) -> None:
        """Frames are read in any order and repeatedly."""
        source = VideoSource(MP4_YUV8)
        sequential = [source.readFrame(index) for index in range(source.frameCount())]

        for index in [0, 1, 2, 40, 3, 59, 0, 40, 40, 2, 3]:
            assert np.array_equal(source.readFrame(index), sequential[index]), index

    def test_readFrame_past_end(self) -> None:
        """There is no frame past the end of the video, whether read in order or by seeking."""
        source = VideoSource(MP4_YUV8)
        for index in range(MP4_YUV8_FRAME_COUNT):
            source.readFrame(index)

        assert source.readFrame(MP4_YUV8_FRAME_COUNT) is None
        assert VideoSource(MP4_YUV8).readFrame(MP4_YUV8_FRAME_COUNT) is None

    def test_pickles_without_capture(self) -> None:
        """An open source pickles as its path and frame times, and the copy reads the same frames in any order."""
        source = VideoSource(MP4_YUV8)
        sequential = [source.readFrame(index) for index in range(source.frameCount())]
        times = source.timestamps

        source = pickle.loads(pickle.dumps(source))

        assert source.path == MP4_YUV8
        assert source.timestamps == times
        assert source.container is None
        for index in [2, 0, 5, 5, 3, 1, 4]:
            assert np.array_equal(source.readFrame(index), sequential[index])

    @ddt.data(
        # The first read is frame 1, which used to read frame 0.
        {"order": [1, 2, 3]},
        # The same frame twice, then back to the start.
        {"order": [1, 1, 0]},
        # The first read is past frame 1, then steps back to it.
        {"order": [2, 3, 1]},
    )
    @ddt.unpack
    def test_fresh_copy_reads_by_index(self, order: list[int]) -> None:
        """A copy that hasn't read anything yet still reads the frames asked for, starting with any frame.

        A fresh capture reports a time of 0 before its first read, which used to look like frame 0
        had just been read, so asking for frame 1 read frame 0 instead.
        """
        expectedFrames = self._decodeFrames(MP4_YUV8)
        source = VideoSource(MP4_YUV8)
        source.frameCount()
        source = pickle.loads(pickle.dumps(source))

        for index in order:
            assert np.array_equal(source.readFrame(index), expectedFrames[index]), index

    def test_dropped_frames_avi_is_missing_frames(self) -> None:
        """Make sure the dropped frames test video has gaps.

        Other tests rely on this video having missing frames.
        """
        container = av.open(DROPPED_FRAMES_AVI)
        headerCount = container.streams.video[0].frames
        container.close()
        source = VideoSource(DROPPED_FRAMES_AVI)
        assert headerCount > source.frameCount()

    @ddt.data(
        # Frames missing from the container, every frame a keyframe.
        {"path": DROPPED_FRAMES_AVI},
        # One keyframe at the start, so every seek must decode forward from frame 0.
        {"path": MP4_YUV8},
        # Several keyframes, so seeks must decode forward from an earlier one.
        {"path": MP4_YUV8_LONG_GOP},
    )
    @ddt.unpack
    def test_reads_correct_frame_however_reached(self, path: str) -> None:
        """A video always fetches the correct frame no matter where we start.

        OpenCV doesn't always land on the frame it seeks to, but our VideoSource wraps OpenCV
        and is designed to work around OpenCV's limitations.

        Check that it works by seeking to many points in the video and doing many
        different lengths of sequential reads. This hopefully simulates both a user
        scrubbing frames and worker pool processing frames.
        """
        # First, decode the frames cleanly and make sure they're unique, so reading the
        # wrong frame can't go unnoticed.
        expectedFrames = self._decodeFrames(path)
        assert len({image.tobytes() for image in expectedFrames}) == len(expectedFrames)

        # Runs of consecutive frames.
        runs: list[tuple[str, Sequence[int]]] = []
        # Full read without any seeks prior to this run.
        runs += [
            ("sequentially read entire file before a bunch of seeks", range(len(expectedFrames)))
        ]
        # Full read, but in reverse direction.
        runs += [
            (f"backward seek to {index} and read 1 frame", [index])
            for index in reversed(range(len(expectedFrames)))
        ]
        # Short runs of 4 frames each, evenly distributed across the file.
        # Different from a full sequential read because there are frame gaps between each run.
        runs += [
            (f"forward seek to {start} and read 4 frames", range(start, min(start + 4, len(expectedFrames))))
            for start in range(0, len(expectedFrames), 5)
        ]
        # Final full read with a potentially unclean internal state.
        runs += [
            ("sequentially read entire file again after a bunch of seeks", range(len(expectedFrames)))
        ]

        # Now, simulate all the runs above through VideoSource.
        source = VideoSource(path)
        actualFrameCount = source.frameCount()
        # pickle + unpickle the VideoSource to simulate being passed into a worker process.
        source = pickle.loads(pickle.dumps(source))
        actualFrames = [
            (name, index, source.readFrame(index))
            for name, run in runs for index in run
        ]

        assert actualFrameCount == len(expectedFrames)
        for name, index, image in actualFrames:
            assert np.array_equal(image, expectedFrames[index]), f"{name}: frame {index} is wrong"

    def test_no_fastPreview(self) -> None:
        """A video has no fast preview."""
        assert VideoSource(MP4_YUV8).fastPreview() is None


@ddt.ddt
class TestOpen(unittest.TestCase):
    """FrameSource.open and canOpen."""

    @ddt.data(
        # A single lossless image.
        {"path": PNG8_SEQUENCE[0],
         "expected_class": ImageSequenceSource,
         "expected_frame_count": 1,
         "expected_frame_size": PNG8_SEQUENCE_DIMENSIONS},
        # A single lossy image.
        {"path": JPG8,
         "expected_class": ImageSequenceSource,
         "expected_frame_count": 1,
         "expected_frame_size": JPG8_DIMENSIONS},
        # A single FITS file, which OpenCV can't read.
        {"path": FITS16_RGGB_SEQUENCE[0],
         "expected_class": ImageSequenceSource,
         "expected_frame_count": 1,
         "expected_frame_size": FITS16_RGGB_SEQUENCE_DIMENSIONS},
        # A single raw camera file.
        {"path": DNG16_RGGB_SEQUENCE[0],
         "expected_class": ImageSequenceSource,
         "expected_frame_count": 1,
         "expected_frame_size": DNG16_RGGB_SEQUENCE_DIMENSIONS},
        # A list of one image opens like the image alone.
        {"path": PNG8_SEQUENCE[:1],
         "expected_class": ImageSequenceSource,
         "expected_frame_count": 1,
         "expected_frame_size": PNG8_SEQUENCE_DIMENSIONS},
        # Several images are a sequence in the order given.
        {"path": PNG8_SEQUENCE,
         "expected_class": ImageSequenceSource,
         "expected_frame_count": len(PNG8_SEQUENCE),
         "expected_frame_size": PNG8_SEQUENCE_DIMENSIONS},
        # Several raw camera files are a sequence.
        {"path": DNG16_RGGB_SEQUENCE,
         "expected_class": ImageSequenceSource,
         "expected_frame_count": len(DNG16_RGGB_SEQUENCE),
         "expected_frame_size": DNG16_RGGB_SEQUENCE_DIMENSIONS},
        # A video.
        {"path": MP4_YUV8,
         "expected_class": VideoSource,
         "expected_frame_count": MP4_YUV8_FRAME_COUNT,
         "expected_frame_size": MP4_YUV8_DIMENSIONS},
        # An image sequence can't contain a video.
        {"path": PNG8_SEQUENCE + [MP4_YUV8], "expected_error": "can't be opened"},
        # A file that is neither an image nor a video.
        {"path": NOT_MEDIA, "expected_error": "can't be opened"},
        # Every file of an image sequence must be an image.
        {"path": PNG8_SEQUENCE + [NOT_MEDIA], "expected_error": "can't be opened"},
        # Nothing chosen.
        {"path": [], "expected_error": "No file was chosen"},
    )
    @ddt.unpack
    def test_open(self,
                  path: str | list[str],
                  expected_class: type[FrameSource] | None = None,
                  expected_frame_count: int | None = None,
                  expected_frame_size: tuple[int, int] | None = None,
                  expected_error: str | None = None) -> None:
        """The chosen files open as the expected kind of source, or fail with expected_error."""
        if(expected_error is not None):
            with self.assertRaisesRegex(ValueError, expected_error):
                FrameSource.open(path)
            return

        source = FrameSource.open(path)

        assert type(source) is expected_class
        assert source.filePaths() == ([path] if isinstance(path, str) else path)
        assert source.frameCount() == expected_frame_count
        assert source.frameSize() == expected_frame_size
        assert source.readFrame(0) is not None
        assert source.readFrame(source.frameCount() - 1) is not None
        assert source.readFrame(source.frameCount()) is None

    @ddt.data(
        # A single image is a one-frame sequence.
        {"choice": PNG8_SEQUENCE[0], "expected_openers": [ImageSequenceSource]},
        # Several images are a sequence.
        {"choice": PNG8_SEQUENCE, "expected_openers": [ImageSequenceSource]},
        # Raw camera files are images.
        {"choice": DNG16_RGGB_SEQUENCE, "expected_openers": [ImageSequenceSource]},
        # A single video is a video.
        {"choice": DROPPED_FRAMES_AVI, "expected_openers": [VideoSource]},
        # A video in a list isn't opened, since only images can be chosen as a list.
        {"choice": [DROPPED_FRAMES_AVI], "expected_openers": []},
        # A video can't be part of an image sequence.
        {"choice": PNG8_SEQUENCE + [DROPPED_FRAMES_AVI], "expected_openers": []},
        # A file that is neither an image nor a video.
        {"choice": NOT_MEDIA, "expected_openers": []},
        # Nothing chosen.
        {"choice": [], "expected_openers": []},
    )
    @ddt.unpack
    def test_canOpen(self, choice: str | list[str], expected_openers: list[type[FrameSource]]) -> None:
        """Each kind of source opens the choices it recognizes by their contents."""
        openers = [source for source in (ImageSequenceSource, VideoSource) if source.canOpen(choice)]

        assert openers == expected_openers
