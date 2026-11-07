"""Paths and properties of the checked-in test inputs, shared by the tests."""
import pathlib

INPUTS_DIR = pathlib.Path(__file__).parent.parent / "features" / "testFiles" / "inputs"
# 8-bit color PNGs, in frame order.
PNG8_SEQUENCE = [str(path) for path in sorted((INPUTS_DIR / "png8_sequence").glob("seq*.png"))]
# The (width, height) of PNG8_SEQUENCE's frames.
PNG8_SEQUENCE_DIMENSIONS = (120, 120)
# png8_sequence's four frames, except the second, which is 4 pixels narrower.
PNG8_MIXED_SIZE_SEQUENCE = [str(path) for path in sorted((INPUTS_DIR / "png8_mixed_size_sequence").glob("seq*.png"))]
# The (width, height) of PNG8_MIXED_SIZE_SEQUENCE's frames, other than the second.
PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS = (120, 120)
# A 16-bit color PNG.
PNG16 = str(INPUTS_DIR / "16bit.png")
# An 8-bit color JPEG.
JPG8 = str(INPUTS_DIR / "test.jpg")
# The (width, height) of JPG8.
JPG8_DIMENSIONS = (191, 125)
# An 8-bit YUV video of Jupiter.
MP4_YUV8 = str(INPUTS_DIR / "Jupiter.mp4")
MP4_YUV8_FRAME_COUNT = 60
# The (width, height) of MP4_YUV8's frames.
MP4_YUV8_DIMENSIONS = (640, 480)
# MP4_YUV8 re-encoded as H.264 with a keyframe every 12 frames, and B-frames in between.
MP4_YUV8_LONG_GOP = str(INPUTS_DIR / "jupiter_long_gop.mp4")
# A video whose header counts 30 frames at 20 fps, but some of the frames are missing.
DROPPED_FRAMES_AVI = str(INPUTS_DIR / "dropped_frames.avi")
# 16-bit FITS images of an RGGB Bayer mosaic, in frame order.
FITS16_RGGB_SEQUENCE = [str(path) for path in sorted((INPUTS_DIR / "fits16_rggb_sequence").glob("rggb_*.fits"))]
# The (width, height) of FITS16_RGGB_SEQUENCE's frames.
FITS16_RGGB_SEQUENCE_DIMENSIONS = (600, 800)
# 16-bit DNG images of an RGGB Bayer mosaic, in frame order.
DNG16_RGGB_SEQUENCE = [str(path) for path in sorted((INPUTS_DIR / "dng16_rggb_sequence").glob("*.dng"))]
# The (width, height) of DNG16_RGGB_SEQUENCE's frames.
# TODO: The true bayer dimension is actually flipped (600x800), so this will
#       need to be fixed once we stop using rawpy to perform demosaic.
DNG16_RGGB_SEQUENCE_DIMENSIONS = (800, 600)
# An 8-bit video of an RGGB Bayer mosaic.
AVI8_RGGB = str(INPUTS_DIR / "avi8_rggb.avi")
# A text file, which is neither an image nor a video.
NOT_MEDIA = str(INPUTS_DIR / "not_an_image.txt")
