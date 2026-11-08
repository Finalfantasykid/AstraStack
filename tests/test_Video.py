"""Tests for Video.py."""
import pathlib
from unittest import mock

import cv2
import numpy as np
import pytest

from FrameSource import FrameSource, ImageSequenceSource
from tests.inputs import PNG8_MIXED_SIZE_SEQUENCE, PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS
from Video import Video, calculateSharpness, loadFrames


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_single_image_getFrame_scales_to_0_1(tmp_path: pathlib.Path, dtype: type) -> None:
    """A single image's frame is float32 on a 0-1 scale, using the full range of the input's bit depth."""
    maxVal = np.iinfo(dtype).max
    image = np.zeros((4, 4, 3), dtype=dtype)
    image[0, 0] = maxVal
    image[1, 1] = maxVal//3
    path = str(tmp_path / "image.png")
    cv2.imwrite(path, image)

    frame = Video(ImageSequenceSource([path])).getFrame(index=0,
                                                        colorMode=Video.COLOR_RGB)

    assert frame.dtype == np.float32
    assert frame.min() == 0
    assert frame.max() == 1
    assert np.array_equal(np.around(frame*maxVal).astype(dtype), image)


@pytest.mark.parametrize("colorMode", [Video.COLOR_AUTO, Video.COLOR_RGB, Video.COLOR_GRAYSCALE, Video.COLOR_RGGB])
@pytest.mark.parametrize("name", ["image.png", "image.jpg"])
def test_single_image_getFrame_matches_readFrame(tmp_path: pathlib.Path, colorMode: int, name: str) -> None:
    """A single image opened as a source reads as its decoded frame, with the color mode applied."""
    rng = np.random.default_rng(seed=0)
    path = str(tmp_path / name)
    cv2.imwrite(path, rng.integers(low=0, high=256, size=(24, 32, 3), dtype=np.uint8))
    image = ImageSequenceSource(path).readFrame(0)
    expectedMode = Video.COLOR_RGB if colorMode == Video.COLOR_AUTO else colorMode
    expected = Video._bgrFromSourceFrame(
        image=image,
        colorMode=expectedMode,
        toDtype=np.float32,
    )

    frame = Video(FrameSource.open(path)).getFrame(index=0,
                                                   colorMode=colorMode)

    assert np.array_equal(frame, expected)


class FakeProgress:
    """Records the calls loadFrames makes to its progress counter."""

    def __init__(self) -> None:
        self.counts: list[tuple[int, int]] = []
        self.extras = 0

    def count(self, i: int, size: int) -> None:
        """Records a counted frame.

        Args:
            i: The frame's position in the chunk.
            size: The chunk's size.
        """
        self.counts.append((i, size))

    def countExtra(self) -> None:
        """Records the end of the chunk."""
        self.extras += 1


def test_loadFrames_skips_other_sizes() -> None:
    """Frames whose size differs are skipped, and every index is counted."""
    progress = FakeProgress()

    frames, sharps, deltas = loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
                                        indices=range(0, 4),
                                        frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                                        colorMode=Video.COLOR_RGB,
                                        progress=progress)

    assert frames == [0, 2, 3]
    assert len(sharps) == 3
    assert len(deltas) == 3
    assert np.array_equal(deltas[0], np.identity(3))
    assert progress.counts == [(i, 4) for i in range(4)]
    assert progress.extras == 1


def test_loadFrames_uses_frame_before_chunk() -> None:
    """A chunk's first delta is taken from the frame before the chunk, as it is when loading all frames."""
    allFrames = loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
                           indices=range(0, 4),
                           frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                           colorMode=Video.COLOR_RGB,
                           progress=FakeProgress())

    frames, sharps, deltas = loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
                                        indices=range(3, 4),
                                        frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                                        colorMode=Video.COLOR_RGB,
                                        progress=FakeProgress())

    assert frames == [3]
    assert sharps == allFrames[1][-1:]
    assert np.array_equal(deltas[0], allFrames[2][-1])


def test_loadFrames_past_end() -> None:
    """Indices past the last frame are counted but load nothing."""
    progress = FakeProgress()

    frames, sharps, deltas = loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
                                        indices=range(3, 7),
                                        frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                                        colorMode=Video.COLOR_RGB,
                                        progress=progress)

    assert frames == [3]
    assert len(sharps) == 1
    assert len(deltas) == 1
    assert len(progress.counts) == 4


def test_bgrFromSourceFrame_rejects_auto() -> None:
    """COLOR_AUTO must be resolved before converting a frame."""
    with pytest.raises(ValueError):
        Video._bgrFromSourceFrame(image=np.zeros((4, 4, 3), dtype=np.uint8), colorMode=Video.COLOR_AUTO)


def test_guessColorMode_guesses_from_first_frame() -> None:
    """Without a frame given, Auto is guessed from the first frame's pixels."""
    assert Video(ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE)).guessColorMode() == Video.COLOR_RGB


def test_loadFrames_measures_16bit_frames_at_8_bits(tmp_path: pathlib.Path) -> None:
    """A 16-bit sequence loads with the same sharpness and deltas as its high bytes saved at 8 bits."""
    paths16 = []
    for index, path in enumerate(PNG8_MIXED_SIZE_SEQUENCE):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.uint16)*256 + 77
        path16 = str(tmp_path / f"seq16_{index}.png")
        cv2.imwrite(path16, image)
        paths16.append(path16)

    loaded8 = loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
                         indices=range(0, 4),
                         frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                         colorMode=Video.COLOR_RGB,
                         progress=FakeProgress())
    loaded16 = loadFrames(source=ImageSequenceSource(paths16),
                          indices=range(0, 4),
                          frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                          colorMode=Video.COLOR_RGB,
                          progress=FakeProgress())

    assert loaded16[0] == loaded8[0]
    assert loaded16[1] == loaded8[1]
    for delta16, delta8 in zip(loaded16[2], loaded8[2]):
        assert np.array_equal(delta16, delta8)


@mock.patch.object(Video, "_bgrFromSourceFrame", wraps=Video._bgrFromSourceFrame)
def test_loadFrames_applies_color_mode_once(bgrFromSourceFrame: mock.MagicMock) -> None:
    """The color mode is applied once to each frame read, including the frame before the chunk."""
    loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
               indices=range(3, 4),
               frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
               colorMode=Video.COLOR_GRBG,
               progress=FakeProgress())

    assert [call.kwargs["colorMode"] for call in bgrFromSourceFrame.call_args_list] == [Video.COLOR_GRBG]*2


@pytest.mark.parametrize("colorMode", [Video.COLOR_RGB, Video.COLOR_GRAYSCALE, Video.COLOR_GRBG])
def test_loadFrames_measures_sharpness_after_color_mode(colorMode: int) -> None:
    """Sharpness is measured on the frame after the color mode is applied."""
    frames, sharps, deltas = loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
                                        indices=range(0, 3),
                                        frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                                        colorMode=colorMode,
                                        progress=FakeProgress())

    for index, sharp in zip(frames, sharps):
        image = Video._bgrFromSourceFrame(
            image=cv2.imread(PNG8_MIXED_SIZE_SEQUENCE[index], cv2.IMREAD_UNCHANGED),
            colorMode=colorMode,
            toDtype=np.uint8,
        )
        assert sharp == calculateSharpness(image)
