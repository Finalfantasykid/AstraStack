"""Tests for Align.py."""
import numpy as np
import pytest

from Align import dilateImg


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("shape", [(40, 40), (40, 40, 3)])
def test_dilateImg_keeps_dtype(dtype: type, shape: tuple) -> None:
    """The output has the input's dtype and shape, for grayscale and color images."""
    img = np.zeros(shape, dtype=dtype)
    img[15:25, 15:25] = 1

    result = dilateImg(img=img, kSize=9, blurSize=5)

    assert result.dtype == dtype
    assert result.shape == img.shape


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_dilateImg_grows_bright_region(dtype: type) -> None:
    """A bright square on a dark grayscale background grows and keeps its brightness."""
    img = np.zeros((40, 40), dtype=dtype)
    img[15:25, 15:25] = 1

    result = dilateImg(img=img, kSize=9, blurSize=5)

    assert np.count_nonzero(result) > np.count_nonzero(img)
    assert result.max() == img.max()
    assert result[0, 0] == 0
