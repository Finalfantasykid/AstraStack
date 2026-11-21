"""Tests for FrameSource.py."""
import os
import pickle
import tempfile
import unittest
from collections.abc import Sequence
from unittest import mock

import cv2
import ddt
import numpy as np

from Constants import AUTO, OFF, UNSPECIFIED, DemosaicAlgorithm, DemosaicParams, MosaicPattern
from FrameSource import CameraRawSource, FrameSource, ImageSequenceSource, OpenCVDemosaicSource, VideoSource
from tests.inputs import (
    AVI8_RGGB,
    DNG16_RGGB_SEQUENCE,
    DNG16_RGGB_SEQUENCE_DIMENSIONS,
    DROPPED_FRAMES_AVI,
    FITS16_RGGB_SEQUENCE,
    FITS16_RGGB_SEQUENCE_DIMENSIONS,
    JPG8,
    JPG8_DIMENSIONS,
    JPG8_NIKON,
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

    def test_readStored_by_index(self) -> None:
        """Frames are read by index in any order, as stored."""
        source = ImageSequenceSource(PNG8_SEQUENCE)

        for index in [2, 0, 3, 1, 1, 2]:
            assert np.array_equal(source.readStored(index), cv2.imread(PNG8_SEQUENCE[index], cv2.IMREAD_UNCHANGED))

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
        source = FrameSource.open([path])

        frame = source.readFrame(0)

        assert frame.dtype == dtype
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert source.frameSize() == (frame.shape[1], frame.shape[0])

    def test_readStored_past_end(self) -> None:
        """There is no frame past the last file."""
        assert ImageSequenceSource(PNG8_SEQUENCE).readStored(len(PNG8_SEQUENCE)) is None

    def test_pickles(self) -> None:
        """A source survives a pickle round trip and reads the same frames."""
        source = ImageSequenceSource(PNG8_SEQUENCE)
        before = source.readStored(1)

        source = pickle.loads(pickle.dumps(source))

        assert source.paths == PNG8_SEQUENCE
        assert np.array_equal(source.readStored(1), before)


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
            return [source.readStored(index) for index in range(source.frameCount())]
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
        assert source.times is None
        # Reading just the first frame does not trigger a full read. This
        # behavior is what makes thumbnail generation quick.
        source.readStored(0)
        assert source.times is None

        assert source.frameCount() == MP4_YUV8_FRAME_COUNT
        source.path = "missing.avi"
        assert source.frameCount() == MP4_YUV8_FRAME_COUNT

    def test_readStored_in_any_order(self) -> None:
        """Frames are read in any order and repeatedly."""
        source = VideoSource(MP4_YUV8)
        sequential = [source.readStored(index) for index in range(source.frameCount())]

        for index in [0, 1, 2, 40, 3, 59, 0, 40, 40, 2, 3]:
            assert np.array_equal(source.readStored(index), sequential[index]), index

    def test_readStored_past_end(self) -> None:
        """There is no frame past the end of the video, whether read in order or by seeking."""
        source = VideoSource(MP4_YUV8)
        for index in range(MP4_YUV8_FRAME_COUNT):
            source.readStored(index)

        assert source.readStored(MP4_YUV8_FRAME_COUNT) is None
        assert VideoSource(MP4_YUV8).readStored(MP4_YUV8_FRAME_COUNT) is None

    def test_pickles_without_capture(self) -> None:
        """An open source pickles as its path and frame times, and the copy reads the same frames in any order."""
        source = VideoSource(MP4_YUV8)
        sequential = [source.readStored(index) for index in range(source.frameCount())]
        times = source.times

        source = pickle.loads(pickle.dumps(source))

        assert source.path == MP4_YUV8
        assert source.times == times
        assert source._capture is None
        for index in [2, 0, 5, 5, 3, 1, 4]:
            assert np.array_equal(source.readStored(index), sequential[index])

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
            assert np.array_equal(source.readStored(index), expectedFrames[index]), index

    def test_dropped_frames_avi_is_missing_frames(self) -> None:
        """Make sure the dropped frames test video has gaps.

        Other tests rely on this video having missing frames.
        """
        capture = cv2.VideoCapture(DROPPED_FRAMES_AVI)
        headerCount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
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
            (name, index, source.readStored(index))
            for name, run in runs for index in run
        ]

        assert actualFrameCount == len(expectedFrames)
        for name, index, image in actualFrames:
            assert np.array_equal(image, expectedFrames[index]), f"{name}: frame {index} is wrong"


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
         "expected_class": CameraRawSource,
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
         "expected_class": CameraRawSource,
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

    def test_jpeg_libraw_opens_is_read_by_opencv(self) -> None:
        """A camera-generated JPEG loads, rather than being mistaken for a raw file because LibRaw opens it."""
        source = FrameSource.open(JPG8_NIKON)

        assert type(source) is ImageSequenceSource
        assert source.readFrame(0) is not None

    @ddt.data(
        # A single image is a one-frame sequence.
        {"choice": PNG8_SEQUENCE[0], "expected_openers": [ImageSequenceSource]},
        # Several images are a sequence.
        {"choice": PNG8_SEQUENCE, "expected_openers": [ImageSequenceSource]},
        # Raw camera files are only camera RAW.
        {"choice": DNG16_RGGB_SEQUENCE, "expected_openers": [CameraRawSource]},
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
        openers = [source for source in (CameraRawSource, ImageSequenceSource, VideoSource) if source.canOpen(choice)]

        assert openers == expected_openers


@ddt.ddt
class TestStaticMethods(unittest.TestCase):
    """The static methods of the frame sources."""

    @ddt.data(
        {"dtype": np.uint8},
        {"dtype": np.uint16},
    )
    @ddt.unpack
    def test_convertDepth_float32_is_0_1_scale(self, dtype: type) -> None:
        """A frame converts to float32 on a 0-1 scale, using the full range of its bit depth."""
        maxVal = np.iinfo(dtype).max
        image = np.array([[0, maxVal//3, maxVal]], dtype=dtype)

        converted = FrameSource.convertDepth(image=image, toDtype=np.float32)

        assert converted.dtype == np.float32
        assert np.array_equal(np.around(converted*maxVal).astype(dtype), image)


@ddt.ddt
class TestDemosaicChoices(unittest.TestCase):
    """setDemosaicChoices and the supported choices, per kind of source."""

    @ddt.data(
        # A gray mosaic (an 8-bit video) is guessed as RGGB under Auto.
        {"path": AVI8_RGGB,
         "chosenPattern": AUTO,
         "expectedResolvedParams": DemosaicParams(MosaicPattern.RGGB, DemosaicAlgorithm.EDGE_AWARE)},
        # A gray mosaic isn't demosaiced when Off.
        {"path": AVI8_RGGB,
         "chosenPattern": OFF,
         "expectedResolvedParams": None},
        # A pattern and an algorithm are honored.
        {"path": AVI8_RGGB,
         "chosenPattern": MosaicPattern.GBRG,
         "chosenAlgorithm": DemosaicAlgorithm.VNG,
         "expectedResolvedParams": DemosaicParams(MosaicPattern.GBRG, DemosaicAlgorithm.VNG)},
        # A color input isn't demosaiced.
        {"path": PNG8_SEQUENCE,
         "chosenPattern": AUTO,
         "expectedResolvedParams": None},
        # A Bayer RAW is demosaiced by LibRaw, which picks the algorithm under Auto.
        {"path": DNG16_RGGB_SEQUENCE,
         "chosenPattern": AUTO,
         "expectedResolvedParams": DemosaicParams(MosaicPattern.RGGB, UNSPECIFIED)},
        # A Bayer RAW is demosaiced with the algorithm chosen, under the pattern it states.
        {"path": DNG16_RGGB_SEQUENCE,
         "chosenPattern": AUTO,
         "chosenAlgorithm": DemosaicAlgorithm.DHT,
         "expectedResolvedParams": DemosaicParams(MosaicPattern.RGGB, DemosaicAlgorithm.DHT)},
    )
    @ddt.unpack
    def test_setDemosaicChoices(
        self,
        path: str | list[str],
        chosenPattern: MosaicPattern | OFF | AUTO,
        expectedResolvedParams: DemosaicParams | None,
        chosenAlgorithm: DemosaicAlgorithm | AUTO = AUTO,
    ) -> None:
        """The choices resolve as the source supports them, and the source keeps the demosaic params."""
        source = FrameSource.open(path)

        resolvedParams = source.setDemosaicChoices(pattern=chosenPattern, algorithm=chosenAlgorithm)

        assert resolvedParams == expectedResolvedParams
        assert source.demosaicParams == expectedResolvedParams

    @ddt.data(
        {"path": AVI8_RGGB, "chosenPattern": AUTO, "expectedDisplayNames": ("RGGB", "Edge-aware")},
        {"path": AVI8_RGGB, "chosenPattern": OFF, "expectedDisplayNames": ("Off", "Off")},
        # LibRaw picks the algorithm.
        {"path": DNG16_RGGB_SEQUENCE, "chosenPattern": AUTO, "expectedDisplayNames": ("RGGB", "LibRaw")},
    )
    @ddt.unpack
    def test_demosaicParamsDisplayNames(
        self,
        path: str | list[str],
        chosenPattern: MosaicPattern | OFF | AUTO,
        expectedDisplayNames: tuple[str, str],
    ) -> None:
        """The source names the demosaic params it resolved."""
        source = FrameSource.open(path)
        source.setDemosaicChoices(pattern=chosenPattern, algorithm=AUTO)

        assert source.demosaicParamsDisplayNames() == expectedDisplayNames

    @ddt.data(
        # Auto doesn't demosaic it.
        {"chosenPattern": AUTO,
         "expectedResolvedParams": None},
        # A chosen pattern demosaics it, Edge-aware.
        {"chosenPattern": MosaicPattern.BGGR,
         "expectedResolvedParams": DemosaicParams(MosaicPattern.BGGR, DemosaicAlgorithm.EDGE_AWARE)},
    )
    @ddt.unpack
    def test_setDemosaicChoices_gray_image(
        self,
        chosenPattern: MosaicPattern | AUTO,
        expectedResolvedParams: DemosaicParams | None,
    ) -> None:
        """A gray image that isn't a mosaic is only demosaiced when asked to."""
        path = os.path.join(self.enterContext(tempfile.TemporaryDirectory()), "gray.png")
        cv2.imwrite(path, cv2.imread(PNG8_SEQUENCE[0], cv2.IMREAD_GRAYSCALE))
        source = FrameSource.open(path)

        assert source.setDemosaicChoices(pattern=chosenPattern, algorithm=AUTO) == expectedResolvedParams

    @ddt.data(
        # The remembered pattern is used when it is in the guessed family.
        {"remembered": MosaicPattern.BGGR, "expectedPattern": MosaicPattern.BGGR},
        {"remembered": MosaicPattern.RGGB, "expectedPattern": MosaicPattern.RGGB},
        # Otherwise the family's first member is.
        {"remembered": MosaicPattern.GBRG, "expectedPattern": MosaicPattern.RGGB},
        {"remembered": MosaicPattern.GRBG, "expectedPattern": MosaicPattern.RGGB},
    )
    @ddt.unpack
    def test_guess_tie_break(self, remembered: MosaicPattern, expectedPattern: MosaicPattern) -> None:
        """The pixel guess knows the family, and the remembered pattern picks the member."""
        demosaicParams = FrameSource.open(AVI8_RGGB).setDemosaicChoices(
            pattern=AUTO,
            algorithm=AUTO,
            rememberedPattern=remembered,
        )

        assert demosaicParams.pattern == expectedPattern

    @mock.patch("cv2.cvtColor", wraps=cv2.cvtColor)
    def test_opencv_fastestDemosaic(self, cvtColor: mock.MagicMock) -> None:
        """OpenCV demosaics with Bilinear instead of the chosen algorithm."""
        source = FrameSource.open(AVI8_RGGB)
        source.setDemosaicChoices(pattern=MosaicPattern.RGGB, algorithm=DemosaicAlgorithm.VNG)

        source.readFrame(0, fastestDemosaic=True)

        cvtColor.assert_any_call(mock.ANY, cv2.COLOR_BAYER_BG2BGR)

    @mock.patch.object(CameraRawSource, "decodeRaw", wraps=CameraRawSource.decodeRaw)
    def test_raw_fastestDemosaic(self, decodeRaw: mock.MagicMock) -> None:
        """LibRaw demosaics with Bilinear instead of the chosen algorithm."""
        source = FrameSource.open(DNG16_RGGB_SEQUENCE)
        source.setDemosaicChoices(pattern=AUTO, algorithm=DemosaicAlgorithm.AHD)

        source.readFrame(0, fastestDemosaic=True)

        assert decodeRaw.call_args.kwargs["algorithm"] == DemosaicAlgorithm.BILINEAR

    def test_opencv_demosaic_codes(self) -> None:
        """Every pattern has a cv2 code for each of OpenCV's algorithms."""
        for name in OpenCVDemosaicSource.OPENCV_DEMOSAIC_CODES.values():
            assert isinstance(getattr(cv2, name), int), name
        assert len(OpenCVDemosaicSource.OPENCV_DEMOSAIC_CODES) == 4*3

    @ddt.data(
        # A 2×2 Bayer pattern is named by its colors.
        {"colorDesc": b"RGBG",
         "rawPattern": np.array([[2, 3], [1, 0]]),
         "expectedPattern": MosaicPattern.BGGR},
        # X-Trans isn't a 2×2 pattern.
        {"colorDesc": b"RGBG",
         "rawPattern": np.zeros((6, 6), dtype=int),
         "expectedPattern": UNSPECIFIED},
    )
    @ddt.unpack
    def test_raw_pattern(
        self,
        colorDesc: bytes,
        rawPattern: np.ndarray,
        expectedPattern: MosaicPattern | UNSPECIFIED,
    ) -> None:
        """A RAW's pattern is read from LibRaw's description of it."""
        raw = mock.Mock(color_desc=colorDesc, raw_pattern=rawPattern)

        assert CameraRawSource.readMosaicPattern(raw) == expectedPattern

    @mock.patch("rawpy.imread")
    def test_raw_demosaic_algorithm(self, imread: mock.MagicMock) -> None:
        """LibRaw picks the algorithm unless one is chosen."""
        import rawpy
        raw = imread.return_value.__enter__.return_value
        raw.postprocess.return_value = np.zeros((2, 2, 3), dtype=np.uint16)

        CameraRawSource.decodeRaw(file="camera.nef")
        CameraRawSource.decodeRaw(file="camera.nef", algorithm=DemosaicAlgorithm.AAHD)

        assert "demosaic_algorithm" not in raw.postprocess.call_args_list[0].kwargs
        assert raw.postprocess.call_args_list[1].kwargs["demosaic_algorithm"] == rawpy.DemosaicAlgorithm.AAHD


class TestCameraRawPreview(unittest.TestCase):
    """CameraRawSource.thumbnail."""

    # A 2×3 BGR image
    IMAGE = np.arange(18, dtype=np.uint8).reshape((2, 3, 3))*10

    @mock.patch("rawpy.imread")
    def test_without_embedded_preview_decodes_at_half_size(self, imread: mock.MagicMock) -> None:
        """A file without a preview is LibRaw's default rendering at half size."""
        import rawpy
        raw = imread.return_value.__enter__.return_value
        raw.extract_thumb.side_effect = rawpy.LibRawNoThumbnailError()
        raw.postprocess.return_value = self.IMAGE[:, :, ::-1]

        preview = CameraRawSource("camera.nef").thumbnail()

        assert np.array_equal(preview, self.IMAGE[:, :, ::-1])
        raw.postprocess.assert_called_once_with(half_size=True, output_bps=8)
