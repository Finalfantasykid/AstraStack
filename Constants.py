"""Enums, constants and small value types shared by modules. Imports nothing of ours, so any module may import it."""
from dataclasses import dataclass
from enum import StrEnum
from typing_extensions import sentinel

# Camera raw formats LibRaw decodes.
LIBRAW_EXTENSIONS = frozenset({
    ".3fr", ".ari", ".arw", ".bay", ".cap", ".cr2", ".cr3", ".crw", ".dcr", ".dng", ".erf", ".fff", ".gpr", ".iiq",
    ".k25", ".kdc", ".mdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf", ".ori", ".pef", ".raf", ".raw", ".rw2",
    ".rwl", ".sr2", ".srf", ".srw", ".x3f",
})

# Still image formats, including all still raw formats.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".fit", ".fits"}) | LIBRAW_EXTENSIONS

# Video formats.
VIDEO_EXTENSIONS = frozenset({
    ".3g2", ".3gp", ".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg", ".ogv", ".ser", ".webm", ".wmv",
})

# The "Auto" option in any drop-down. This is a signal from the user that they
# defer to the app to decide, rather than representing a concrete option.
AUTO = sentinel("AUTO")

# An internal signal from the app that it defers to lower-level functions and
# 3rd party libraries to decide. This is NOT selectable by the user, and needs
# to be distinct from AUTO because otherwise we wouldn't know if AUTO meant the
# decision still needs to be resolved by the app or should be deferred to the
# library.
UNSPECIFIED = sentinel("UNSPECIFIED")

# The "Off" option of the Pattern drop-down: the user's choice not to demosaic the frames.
# Its repr is the display name, like the values of the enums below.
OFF = sentinel("OFF", repr="Off")

class MosaicPattern(StrEnum):
    """The 2×2 color filter pattern of a mosaic. The values are display names."""
    RGGB = "RGGB"
    BGGR = "BGGR"
    GRBG = "GRBG"
    GBRG = "GBRG"

class DemosaicAlgorithm(StrEnum):
    """The demosaic algorithms OpenCV and LibRaw offer in their standard builds. The values are display names."""
    EDGE_AWARE = "Edge-aware"
    BILINEAR = "Bilinear"
    VNG = "VNG"
    PPG = "PPG"
    AHD = "AHD"
    DCB = "DCB"
    DHT = "DHT"
    AAHD = "AAHD"

@dataclass(frozen=True)
class DemosaicParams:
    """The resolved demosaic params: the frames are a mosaic of this pattern, demosaiced by this algorithm."""
    pattern: MosaicPattern | UNSPECIFIED
    algorithm: DemosaicAlgorithm | UNSPECIFIED
