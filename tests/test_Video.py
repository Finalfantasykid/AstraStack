"""Tests for Video.py."""
import pathlib

import cv2
import numpy as np
import pytest

from Video import Video


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_getFrame_scales_to_0_1(tmp_path: pathlib.Path, dtype: type) -> None:
    """getFrame returns float32 on a 0-1 scale, using the full range of the input's bit depth."""
    maxVal = np.iinfo(dtype).max
    image = np.zeros((4, 4, 3), dtype=dtype)
    image[0, 0] = maxVal
    image[1, 1] = maxVal//3
    path = str(tmp_path / "image.png")
    cv2.imwrite(path, image)

    frame = Video().getFrame(file=None, frame=path, colorMode=Video.COLOR_RGB)

    assert frame.dtype == np.float32
    assert frame.min() == 0
    assert frame.max() == 1
    assert np.array_equal(np.around(frame*maxVal).astype(dtype), image)
