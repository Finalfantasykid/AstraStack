from __future__ import annotations
from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
rawpy = lazy("rawpy")
import bisect
import dataclasses
import os
from abc import ABC, abstractmethod
from functools import cached_property
from typing import NamedTuple, override
from natsort import natsorted, ns
from Constants import (
    AUTO,
    LIBRAW_EXTENSIONS,
    OFF,
    UNSPECIFIED,
    DemosaicAlgorithm,
    DemosaicParams,
    MosaicPattern,
)

class FrameSource(ABC):
    """A sequence of frames stored in one or more files, addressed by integer index.

    Sources are sent to worker processes, so they must pickle as just what identifies their
    frames and open files lazily wherever they are read.
    """

    # The display name of the library that demosaics the frames.
    DEMOSAIC_LIBRARY_NAME: str

    # The algorithm Auto demosaics with; UNSPECIFIED lets the library choose.
    DEFAULT_DEMOSAIC_ALGORITHM: DemosaicAlgorithm | UNSPECIFIED

    # The library's fastest demosaic algorithm, whatever its quality.
    FASTEST_DEMOSAIC_ALGORITHM: DemosaicAlgorithm

    # The demosaic params to actually send to the demosaicing library. None to disable demosaicing.
    demosaicParams: DemosaicParams | None = None

    @classmethod
    def open(cls, path: str | list[str]) -> FrameSource:
        """Opens the chosen file or files as the one kind of source that can open them.

        Args:
            path: A file, or a list of files in any order.

        Returns:
            The source.

        Raises:
            ValueError: If no file was chosen or no kind of source can open the files; the
                message is meant for the user.
        """
        if(len(path) == 0):
            raise ValueError("No file was chosen.")
        for source in (CameraRawSource, ImageSequenceSource, VideoSource):
            if(source.canOpen(path)):
                return source(path)
        raise ValueError("The chosen files can't be opened. Choose one video, or one or more images.")

    @staticmethod
    @abstractmethod
    def canOpen(path: str | list[str]) -> bool:
        """Returns whether this kind of source can open the chosen file or files.

        Files are recognized by their contents rather than their extensions. Kinds of source
        don't overlap, so at most one can open any choice.

        Args:
            path: A file, or a list of files in any order.

        Returns:
            Whether this kind of source can open them.
        """

    @abstractmethod
    def filePaths(self) -> list[str]:
        """Returns the files the source reads.

        Returns:
            The files, in frame order.
        """

    @abstractmethod
    def frameCount(self) -> int:
        """Returns the number of frames in the source.

        Returns:
            The number of frames; indices run from 0.
        """

    @abstractmethod
    def readFrame(self, index: int, fastestDemosaic: bool = False) -> np.ndarray | None:
        """Reads one frame as BGR, demosaiced as the source's demosaic params say.

        Args:
            index: The frame's index.
            fastestDemosaic:
                If True, demosaic with the fastest algorithm instead of the one set on ``self.demosaicParams``.

        Returns:
            The frame in BGR order at its stored depth (uint8 or uint16), or None if the source
            has no readable frame at index.
        """

    @abstractmethod
    def frameSize(self) -> tuple[int, int]:
        """Returns the size of the frames.

        Returns:
            The (width, height) of the first frame.
        """

    @abstractmethod
    def thumbnail(self) -> np.ndarray:
        """Makes a display-ready thumbnail of the source, leaving its demosaic params as they are.

        The thumbnail is shown as sRGB as it is, so it only needs to look close to the first frame, as a camera's own
        rendering does.

        Returns:
            The thumbnail as uint8 RGB, possibly at a reduced size.
        """

    @staticmethod
    def convertDepth(image: np.ndarray, toDtype: type | None = None) -> np.ndarray:
        """Converts a frame read from a source to another depth.

        Args:
            image: A frame as readFrame returns it.
            toDtype: The output's dtype: np.float32 for a 0-1 scale, np.uint8 for 8 bits
                (16-bit frames keep their high byte), or None to keep the input's dtype.

        Returns:
            The frame at the new depth.

        Raises:
            ValueError: If the frame's dtype or toDtype isn't supported.
        """
        if image.dtype not in (np.uint8, np.uint16):
            raise ValueError(f"Unsupported frame dtype: {image.dtype}")
        if toDtype not in (None, np.float32, np.uint8):
            raise ValueError(f"Unsupported toDtype: {toDtype}")
        if toDtype == np.float32:
            return image.astype(np.float32) / np.iinfo(image.dtype).max
        if toDtype == np.uint8 and image.dtype == np.uint16:
            return (image >> 8).astype(np.uint8)
        return image

    @abstractmethod
    def supportedMosaicPatterns(self) -> list[MosaicPattern | OFF]:
        """Returns the patterns the source supports; OFF if it can be left mosaiced."""

    @abstractmethod
    def supportedDemosaicAlgorithms(self) -> list[DemosaicAlgorithm]:
        """Returns the algorithms the source supports."""

    @abstractmethod
    def autoDemosaicParams(self, rememberedPattern: MosaicPattern | None = None) -> DemosaicParams | None:
        """Returns the demosaic params Auto resolves to.

        Args:
            rememberedPattern: The pattern the user last picked, which breaks a pixel guess's tie; without one, the
                guessed family's first member is used.

        Returns:
            The demosaic params, or None if Auto doesn't demosaic the frames.
        """

    def dropUnsupportedDemosaicChoices(
        self,
        pattern: MosaicPattern | OFF | AUTO,
        algorithm: DemosaicAlgorithm | AUTO,
    ) -> tuple[MosaicPattern | OFF | AUTO, DemosaicAlgorithm | AUTO]:
        """Reverts each demosaic choice the source doesn't support to AUTO.

        Args:
            pattern: The pattern, OFF, or AUTO.
            algorithm: The algorithm, or AUTO.

        Returns:
            The pattern and algorithm, each AUTO if the source doesn't support it.
        """
        return (
            pattern if pattern in self.supportedMosaicPatterns() else AUTO,
            algorithm if algorithm in self.supportedDemosaicAlgorithms() else AUTO,
        )

    def setDemosaicChoices(
        self,
        pattern: MosaicPattern | OFF | AUTO,
        algorithm: DemosaicAlgorithm | AUTO,
        rememberedPattern: MosaicPattern | None = None,
    ) -> DemosaicParams | None:
        """Configures this source to read frames with the user's demosaic choices.

        A choice the source doesn't support is taken as AUTO.

        Args:
            pattern: The pattern, OFF not to demosaic, or AUTO.
            algorithm: The algorithm, or AUTO. Ignored if the frames aren't demosaiced.
            rememberedPattern: The pattern the user last picked, which breaks a pixel guess's tie.

        Returns:
            The resolved demosaic params; None if the frames aren't demosaiced.
        """
        pattern, algorithm = self.dropUnsupportedDemosaicChoices(pattern, algorithm)
        autoParams = self.autoDemosaicParams(rememberedPattern)
        if(pattern is AUTO):
            pattern = autoParams.pattern if autoParams else OFF
        if(pattern is OFF):
            # Disable demosaicing if the user has turned it off manually, or no pattern was automatically detected.
            self.demosaicParams = None
        else:
            # The default algorithm if the user picked a pattern Auto didn't find
            autoAlgorithm = autoParams.algorithm if autoParams else self.DEFAULT_DEMOSAIC_ALGORITHM
            self.demosaicParams = DemosaicParams(pattern, autoAlgorithm if algorithm is AUTO else algorithm)
        return self.demosaicParams

    def getDemosaicParams(self, fastestDemosaic: bool) -> DemosaicParams | None:
        """Returns the demosaic params readFrame demosaics with.

        Args:
            fastestDemosaic: Whether to swap in the fastest algorithm, if the source supports it.

        Returns:
            The demosaic params, or None if the frames aren't demosaiced.
        """
        if(
            fastestDemosaic
            and self.demosaicParams is not None
            and self.FASTEST_DEMOSAIC_ALGORITHM in self.supportedDemosaicAlgorithms()
        ):
            return dataclasses.replace(self.demosaicParams, algorithm=self.FASTEST_DEMOSAIC_ALGORITHM)
        return self.demosaicParams

    def demosaicParamsDisplayNames(self) -> tuple[str, str]:
        """Names the demosaic pattern and algorithm the frames are read with, for display.

        Returns:
            The names of the pattern and algorithm, e.g. ("RGGB", "Edge-aware"). Both are "Off" if the frames aren't
            demosaiced. The library's name stands for a choice it makes itself.
        """
        if(self.demosaicParams is None):
            return ("Off", "Off")
        pattern, algorithm = (
            self.DEMOSAIC_LIBRARY_NAME if choice is UNSPECIFIED else str(choice)
            for choice in (self.demosaicParams.pattern, self.demosaicParams.algorithm)
        )
        return (pattern, algorithm)

    def close(self) -> None:
        """Releases any open file handles. The source reopens them if it is read again."""

class OpenCVDemosaicSource(FrameSource):
    """A source whose stored frames may be a mosaic, which OpenCV demosaics.

    Auto demosaics gray frames that the pixel guess finds a mosaic in, with Edge-aware.
    """

    DEMOSAIC_LIBRARY_NAME = "OpenCV"
    DEFAULT_DEMOSAIC_ALGORITHM = DemosaicAlgorithm.EDGE_AWARE
    FASTEST_DEMOSAIC_ALGORITHM = DemosaicAlgorithm.BILINEAR

    # The cv2 color conversion code of each pattern and algorithm OpenCV
    # demosaics. Stored as strings to avoid triggering the lazy cv2 import.
    OPENCV_DEMOSAIC_CODES = {
        DemosaicParams(MosaicPattern.RGGB, DemosaicAlgorithm.BILINEAR): "COLOR_BAYER_BG2BGR",
        DemosaicParams(MosaicPattern.GRBG, DemosaicAlgorithm.BILINEAR): "COLOR_BAYER_GB2BGR",
        DemosaicParams(MosaicPattern.GBRG, DemosaicAlgorithm.BILINEAR): "COLOR_BAYER_GR2BGR",
        DemosaicParams(MosaicPattern.BGGR, DemosaicAlgorithm.BILINEAR): "COLOR_BAYER_RG2BGR",
        DemosaicParams(MosaicPattern.RGGB, DemosaicAlgorithm.EDGE_AWARE): "COLOR_BAYER_BG2BGR_EA",
        DemosaicParams(MosaicPattern.GRBG, DemosaicAlgorithm.EDGE_AWARE): "COLOR_BAYER_GB2BGR_EA",
        DemosaicParams(MosaicPattern.GBRG, DemosaicAlgorithm.EDGE_AWARE): "COLOR_BAYER_GR2BGR_EA",
        DemosaicParams(MosaicPattern.BGGR, DemosaicAlgorithm.EDGE_AWARE): "COLOR_BAYER_RG2BGR_EA",
        DemosaicParams(MosaicPattern.RGGB, DemosaicAlgorithm.VNG): "COLOR_BAYER_BG2BGR_VNG",
        DemosaicParams(MosaicPattern.GRBG, DemosaicAlgorithm.VNG): "COLOR_BAYER_GB2BGR_VNG",
        DemosaicParams(MosaicPattern.GBRG, DemosaicAlgorithm.VNG): "COLOR_BAYER_GR2BGR_VNG",
        DemosaicParams(MosaicPattern.BGGR, DemosaicAlgorithm.VNG): "COLOR_BAYER_RG2BGR_VNG",
    }

    # The pixel guess can tell a mosaic's family but not which member: swapping red and blue looks the same. Unless
    # the user last picked the other member, each family's first is used.
    MOSAIC_FAMILIES = ((MosaicPattern.RGGB, MosaicPattern.BGGR), (MosaicPattern.GRBG, MosaicPattern.GBRG))

    class FrameFacts(NamedTuple):
        """Hints about the inputs derived from one frame."""
        isGray: bool  # Whether its channels are equal, so it may be a mosaic
        is8Bit: bool  # Whether it is stored at 8 bits
        mosaicFamily: tuple[MosaicPattern, MosaicPattern] | None  # The family guessMosaicFamily finds, if any

    def __getstate__(self) -> dict[str, object]:
        """Pickles everything but the cached first frame, for memory efficiency.

        Workers don't use the frame, and a pickled source is copied into every worker job, which for large frames
        would add up to a lot of memory. Deep copies leave it out too.
        """
        state = self.__dict__.copy()
        state.pop("firstFrame", None)
        return state

    @abstractmethod
    def readStored(self, index: int) -> np.ndarray | None:
        """Reads one frame as stored.

        Args:
            index: The frame's index.

        Returns:
            The frame in BGR order, or with a single channel for gray or Bayer mosaic frames,
            at its stored depth (uint8 or uint16); or None if the source has no readable frame
            at index.
        """

    @cached_property
    def firstFrame(self) -> np.ndarray | None:
        """The first frame as readStored returns it, read once for the thumbnail and the facts; close releases it."""
        return self.readStored(0)

    @cached_property
    def firstFrameFacts(self) -> OpenCVDemosaicSource.FrameFacts:
        """What the first frame says about the source, derived the first time it is needed."""
        image = self.firstFrame
        if(image is None):
            # Without a first frame to judge by, the frames are read as color, as they are when no mosaic is found
            return OpenCVDemosaicSource.FrameFacts(isGray=False, is8Bit=False, mosaicFamily=None)
        isGray = OpenCVDemosaicSource.isGray(image)
        return OpenCVDemosaicSource.FrameFacts(
            isGray=isGray,
            is8Bit=image.dtype == np.uint8,
            mosaicFamily=OpenCVDemosaicSource.guessMosaicFamily(image) if isGray else None,
        )

    @staticmethod
    def isGray(image: np.ndarray) -> bool:
        """Returns whether a frame has equal channels."""
        return image.ndim == 2 or (np.array_equal(image[:, :, 0], image[:, :, 1]) and
                                   np.array_equal(image[:, :, 1], image[:, :, 2]))

    @override
    def supportedMosaicPatterns(self) -> list[MosaicPattern | OFF]:
        """A gray frame may be a mosaic of any 2×2 pattern, a color one is not."""
        return [OFF, *MosaicPattern] if self.firstFrameFacts.isGray else []

    @override
    def supportedDemosaicAlgorithms(self) -> list[DemosaicAlgorithm]:
        """OpenCV's algorithms for gray frames; VNG only works on 8-bit frames."""
        first = self.firstFrameFacts
        if(not first.isGray):
            return []
        vng = [DemosaicAlgorithm.VNG] if first.is8Bit else []
        return [DemosaicAlgorithm.BILINEAR, DemosaicAlgorithm.EDGE_AWARE, *vng]

    @override
    def autoDemosaicParams(self, rememberedPattern: MosaicPattern | None = None) -> DemosaicParams | None:
        """Gray frames are a mosaic when the pixel guess finds one, of the family member the user last picked or else
        the family's first (RGGB or GRBG); the algorithm is Edge-aware.
        """
        family = self.firstFrameFacts.mosaicFamily
        if(family is None):
            return None
        pattern = rememberedPattern if rememberedPattern in family else family[0]
        return DemosaicParams(pattern, self.DEFAULT_DEMOSAIC_ALGORITHM)

    @staticmethod
    def guessMosaicFamily(image: np.ndarray) -> tuple[MosaicPattern, MosaicPattern] | None:
        """Guesses from a gray frame's pixels whether it is a mosaic.

        A very simple guess, far from perfect: demosaicing a mosaic reveals a pattern. It is only told apart as
        RGGB/BGGR or GRBG/GBRG.

        Args:
            image: A frame with equal channels, as readStored returns it.

        Returns:
            The family the pattern is in, or None if the frame isn't a mosaic.
        """
        rggb, grbg, gbrg, bggr = (
            OpenCVDemosaicSource.demosaic(
                img=image,
                demosaicParams=DemosaicParams(pattern, DemosaicAlgorithm.EDGE_AWARE),
            )
            for pattern in (MosaicPattern.RGGB, MosaicPattern.GRBG, MosaicPattern.GBRG, MosaicPattern.BGGR)
        )
        diff1 = np.average(cv2.absdiff(cv2.cvtColor(rggb, cv2.COLOR_BGR2GRAY), cv2.cvtColor(bggr, cv2.COLOR_BGR2GRAY)))
        diff2 = np.average(cv2.absdiff(cv2.cvtColor(grbg, cv2.COLOR_BGR2GRAY), cv2.cvtColor(gbrg, cv2.COLOR_BGR2GRAY)))
        diff = 0
        if(diff1 > 0.0001 and diff2 > 0.0001):
            diff = abs(diff1 - diff2)/min(diff1, diff2)
        if(diff <= 0.2):
            return None
        # Probably uses a bayer pattern, but which family...
        rggbDiff = np.average(cv2.absdiff(rggb[:,:,0], rggb[:,:,2]))
        grbgDiff = np.average(cv2.absdiff(grbg[:,:,0], grbg[:,:,2]))
        return OpenCVDemosaicSource.MOSAIC_FAMILIES[0 if rggbDiff >= grbgDiff else 1]

    @staticmethod
    def demosaic(img: np.ndarray, demosaicParams: DemosaicParams | None) -> np.ndarray:
        """Demosaics a frame with OpenCV, keeping its depth.

        VNG demosaicing is only done on uint8 frames; deeper frames are demosaiced edge-aware.

        Args:
            img: The frame as readStored returns it; a mosaic if demosaicParams isn't None.
            demosaicParams: The pattern and algorithm, which are not UNSPECIFIED; None to leave the frame as it is.

        Returns:
            The frame as BGR.
        """
        if(demosaicParams is None):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        h, w = img.shape[:2]
        if(img.ndim == 3):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if(demosaicParams.algorithm == DemosaicAlgorithm.VNG and img.dtype != np.uint8):
            demosaicParams = DemosaicParams(demosaicParams.pattern, DemosaicAlgorithm.EDGE_AWARE)
        code = getattr(cv2, OpenCVDemosaicSource.OPENCV_DEMOSAIC_CODES[demosaicParams])
        if(demosaicParams.algorithm == DemosaicAlgorithm.VNG):
            img = cv2.copyMakeBorder(img, 4, 2, 2, 2, cv2.BORDER_REPLICATE)
        else:
            img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
        return cv2.cvtColor(img, code)[2:h+2, 2:w+2]

    @override
    def readFrame(self, index: int, fastestDemosaic: bool = False) -> np.ndarray | None:
        """The stored frame is demosaiced with OpenCV."""
        image = self.readStored(index)
        if(image is None):
            return None
        return OpenCVDemosaicSource.demosaic(img=image, demosaicParams=self.getDemosaicParams(fastestDemosaic))

    @override
    def close(self) -> None:
        """Also releases the cached first frame, so a loaded source doesn't keep it; it's read again if needed."""
        self.__dict__.pop("firstFrame", None)

    @override
    def thumbnail(self) -> np.ndarray:
        """The first frame, demosaiced as Auto resolves."""
        image = OpenCVDemosaicSource.demosaic(img=self.firstFrame, demosaicParams=self.autoDemosaicParams())
        image = FrameSource.convertDepth(image=image, toDtype=np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

class ImageSequenceSource(OpenCVDemosaicSource):
    """A list of image files, one frame per file."""

    def __init__(self, path: str | list[str]) -> None:
        """Initializes the source.

        Args:
            path: An image file, or a list of image files (in any order).
        """
        self.paths = [path] if isinstance(path, str) else natsorted(path, alg=ns.IGNORECASE)

    @staticmethod
    @override
    def canOpen(path: str | list[str]) -> bool:
        """Accepts an image file, or a list of one or more image files, but not raw camera files.

        Each file must be recognized, without decoding its pixels, by one of the readers
        readStored decodes with, in its order: an image format OpenCV reads, then a FITS file
        with image data.
        """
        paths = [path] if isinstance(path, str) else path
        for file in paths:
            if(CameraRawSource.isRaw(file)):
                return False
            if(cv2.haveImageReader(file)):
                continue
            try:
                from astropy.io import fits
                with fits.open(file) as hdus:
                    if(any(hdu.header.get("NAXIS", 0) > 0 for hdu in hdus)):
                        continue
            except Exception:
                pass
            return False
        return len(paths) > 0

    @override
    def filePaths(self) -> list[str]:
        return self.paths

    @override
    def frameCount(self) -> int:
        return len(self.paths)

    @override
    def readStored(self, index: int) -> np.ndarray | None:
        """FITS files are scaled to 16 bits."""
        if(index >= len(self.paths)):
            return None
        return ImageSequenceSource._readImageFile(self.paths[index])

    @staticmethod
    def _readImageFile(file: str) -> np.ndarray:
        """Decodes an image file as an image OpenCV reads, else FITS.

        Args:
            file: The image file.

        Returns:
            The image. An image OpenCV reads keeps its stored depth and channels, and a FITS
            image is stretched to the full 16-bit range.
        """
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if image is not None:
            return image
        from astropy.io import fits

        image = fits.getdata(file)
        maxVal = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1
        maxOut = np.iinfo(np.uint16).max
        image = (image.astype(np.float32) / maxVal) * maxOut
        cv2.normalize(image, image, alpha=0, beta=maxOut, norm_type=cv2.NORM_MINMAX)
        return image.astype(np.uint16)

    @override
    def frameSize(self) -> tuple[int, int]:
        height, width = self.readStored(0).shape[:2]
        return (width, height)

class CameraRawSource(FrameSource):
    """A list of camera RAW files, one frame per file, decoded and demosaiced by LibRaw."""

    DEMOSAIC_LIBRARY_NAME = "LibRaw"
    DEFAULT_DEMOSAIC_ALGORITHM = UNSPECIFIED
    FASTEST_DEMOSAIC_ALGORITHM = DemosaicAlgorithm.BILINEAR

    # LibRaw's names of the algorithms it offers in its standard build
    LIBRAW_DEMOSAIC_ALGORITHMS = {
        DemosaicAlgorithm.BILINEAR: "LINEAR",
        DemosaicAlgorithm.VNG: "VNG",
        DemosaicAlgorithm.PPG: "PPG",
        DemosaicAlgorithm.AHD: "AHD",
        DemosaicAlgorithm.DCB: "DCB",
        DemosaicAlgorithm.DHT: "DHT",
        DemosaicAlgorithm.AAHD: "AAHD",
    }

    class FrameFacts(NamedTuple):
        """Hints about the inputs derived from one frame."""
        sensorPattern: MosaicPattern | UNSPECIFIED | None  # The sensor's pattern, as readMosaicPattern reads it

    def __init__(self, path: str | list[str]) -> None:
        """Initializes the source.

        Args:
            path: A raw camera file, or a list of raw camera files (in any order).
        """
        self.paths = [path] if isinstance(path, str) else natsorted(path, alg=ns.IGNORECASE)

    @staticmethod
    def isRaw(file: str) -> bool:
        """Returns whether LibRaw recognizes a file, without decoding its pixels.

        Only a file with a raw camera extension is tried: LibRaw opens some other files, such as JPEGs from Nikon
        cameras, then fails to decode them.

        Args:
            file: Path of the file.

        Returns:
            Whether it is a raw camera file.
        """
        if(os.path.splitext(file)[1].lower() not in LIBRAW_EXTENSIONS):
            return False
        try:
            with rawpy.imread(file):
                return True
        except Exception:
            return False

    @staticmethod
    @override
    def canOpen(path: str | list[str]) -> bool:
        """Accepts a raw camera file, or a list of one or more of them."""
        paths = [path] if isinstance(path, str) else path
        return len(paths) > 0 and all(CameraRawSource.isRaw(file) for file in paths)

    @override
    def filePaths(self) -> list[str]:
        return self.paths

    @override
    def frameCount(self) -> int:
        return len(self.paths)

    @override
    def frameSize(self) -> tuple[int, int]:
        height, width = CameraRawSource.decodeRaw(file=self.paths[0]).shape[:2]
        return (width, height)

    @cached_property
    def firstFrameFacts(self) -> CameraRawSource.FrameFacts:
        """What the first file says about the source, read from its metadata without decoding."""
        with rawpy.imread(self.paths[0]) as raw:
            return CameraRawSource.FrameFacts(sensorPattern=CameraRawSource.readMosaicPattern(raw))

    @staticmethod
    def readMosaicPattern(raw: rawpy.RawPy) -> MosaicPattern | UNSPECIFIED | None:
        """Reads the sensor's pattern from LibRaw's own description of it.

        Args:
            raw: The opened raw camera file.

        Returns:
            None if the sensor isn't a mosaic (monochrome, linear DNG, Foveon), UNSPECIFIED if its pattern isn't a
            2×2 one (X-Trans, 4-color sensors), else the pattern, e.g. RGGB.
        """
        if(raw.raw_pattern is None):
            return None
        letters = "".join(chr(raw.color_desc[i]) for i in raw.raw_pattern.flatten())
        return MosaicPattern.__members__.get(letters, UNSPECIFIED)

    @override
    def supportedMosaicPatterns(self) -> list[MosaicPattern | OFF]:
        """LibRaw reads the pattern from the file, and demosaics any mosaic, which can't be left alone."""
        return []

    @override
    def supportedDemosaicAlgorithms(self) -> list[DemosaicAlgorithm]:
        """LibRaw has the algorithms for 2×2 Bayer mosaics."""
        isBayer = isinstance(self.firstFrameFacts.sensorPattern, MosaicPattern)
        return list(CameraRawSource.LIBRAW_DEMOSAIC_ALGORITHMS) if isBayer else []

    @override
    def autoDemosaicParams(self, rememberedPattern: MosaicPattern | None = None) -> DemosaicParams | None:
        """The frames are a mosaic when LibRaw says so, demosaiced with the algorithm LibRaw picks."""
        pattern = self.firstFrameFacts.sensorPattern
        return None if pattern is None else DemosaicParams(pattern, self.DEFAULT_DEMOSAIC_ALGORITHM)

    @override
    def readFrame(self, index: int, fastestDemosaic: bool = False) -> np.ndarray | None:
        """LibRaw demosaics the frame, with the demosaic params' algorithm if they have one, else as it chooses."""
        if(index >= len(self.paths)):
            return None
        demosaicParams = self.getDemosaicParams(fastestDemosaic)
        algorithm = UNSPECIFIED if demosaicParams is None else demosaicParams.algorithm
        return CameraRawSource.decodeRaw(file=self.paths[index], algorithm=algorithm)

    @override
    def thumbnail(self) -> np.ndarray:
        """The first file's embedded preview, turned upright as LibRaw would turn the image; or else, for a file
        without an RGB preview, LibRaw's default rendering at half size.
        """
        with rawpy.imread(self.paths[0]) as raw:
            preview = CameraRawSource.readEmbeddedPreview(raw)
            if(preview is None):
                # sRGB with LibRaw's gamma, already upright
                return raw.postprocess(half_size=True, output_bps=8)
            flip = raw.sizes.flip
        # Map LibRaw rotation codes to OpenCV ones.
        rotation = {3: cv2.ROTATE_180, 5: cv2.ROTATE_90_COUNTERCLOCKWISE, 6: cv2.ROTATE_90_CLOCKWISE}.get(flip)
        return preview if rotation is None else cv2.rotate(preview, rotation)

    @staticmethod
    def readEmbeddedPreview(raw: rawpy.RawPy) -> np.ndarray | None:
        """Reads a raw camera file's embedded preview, as stored, without turning it upright.

        Args:
            raw: The opened raw camera file.

        Returns:
            The preview as uint8 RGB, or None if the file has no RGB preview LibRaw can extract.
        """
        try:
            thumb = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            return None
        if(thumb.format == rawpy.ThumbFormat.BITMAP):
            # LibRaw also extracts some gray bitmaps
            return thumb.data if thumb.data.shape[2] == 3 else None
        # Ignores any orientation tag, since not every camera tags its preview's orientation
        flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(thumb.data, np.uint8), flags), cv2.COLOR_BGR2RGB)

    @staticmethod
    def decodeRaw(file: str, algorithm: DemosaicAlgorithm | UNSPECIFIED = UNSPECIFIED) -> np.ndarray:
        """Decodes a raw camera file with LibRaw.

        Args:
            file: Path of the file.
            algorithm: The demosaic algorithm; UNSPECIFIED lets LibRaw choose.

        Returns:
            The image as 16-bit BGR.
        """
        options = {}
        if(algorithm is not UNSPECIFIED):
            libRawAlgorithmName = CameraRawSource.LIBRAW_DEMOSAIC_ALGORITHMS[algorithm]
            options["demosaic_algorithm"] = rawpy.DemosaicAlgorithm[libRawAlgorithmName]
        with rawpy.imread(file) as raw:
            img = raw.postprocess(output_bps=16, **options)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

class VideoSource(OpenCVDemosaicSource):
    """A video file read with cv2.VideoCapture.

    Frame indexes always represent unique frames in the data stream, and that's
    all consumers should care about. Under the hood, we map each unique frame
    to their time and later use that time for seeking to the frame.
    """

    def __init__(self, path: str) -> None:
        """Initializes the source.

        Args:
            path: The video file.
        """
        self.path = path
        self.times: list[float] | None = None
        self._capture: cv2.VideoCapture | None = None
        self.nextPosition: int | None = 0

    def __getstate__(self) -> dict[str, object]:
        """Pickles everything but the open capture, so each process opens its own file descriptor."""
        state = super().__getstate__()
        del state["_capture"]
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restores a pickled source, which opens its capture when first read."""
        self.__dict__.update(state)
        self._capture = None
        self.nextPosition = 0

    @staticmethod
    @override
    def canOpen(path: str | list[str]) -> bool:
        """Accepts a single file OpenCV opens and reads at least two frames from.

        One-frame videos are rejected, and left to other FrameSource implementations to handle.
        """
        if(not isinstance(path, str)):
            return False
        capture = cv2.VideoCapture(path)
        try:
            return capture.isOpened() and capture.read()[0] and capture.grab()
        finally:
            capture.release()

    @override
    def filePaths(self) -> list[str]:
        return [self.path]

    @property
    def capture(self) -> cv2.VideoCapture:
        """The capture, opened at the first frame if it isn't open."""
        if(self._capture is None):
            self._capture = cv2.VideoCapture(self.path)
            self.nextPosition = 0
        return self._capture

    def _frameTimes(self) -> list[float]:
        """Returns the cached times of the frames.

        The times are read by grabbing every frame sequentially the first time.

        Returns:
            The frame times in milliseconds, in frame order.
        """
        if(self.times is None):
            capture = cv2.VideoCapture(self.path)
            try:
                times = []
                while(capture.grab()):
                    times.append(capture.get(cv2.CAP_PROP_POS_MSEC))
                self.times = times
            finally:
                capture.release()
        return self.times

    @override
    def frameCount(self) -> int:
        """The count comes from grabbing every frame, since CAP_PROP_FRAME_COUNT is only the container's estimate."""
        return len(self._frameTimes())

    def _seekBefore(self, index: int) -> None:
        """Seeks to the frame, or to an earlier frame that reading forward reaches it from.

        Usage: To find your desired frame at `index`, run _seekBefore() then sequentially
        read frames until `self.nextPosition == index + 1`.

        A naive approach that simply seeks to the time given by the index is NOT
        guaranteed to make OpenCV return the expected frame.  Often, a seek + read() will
        give you 2, 3 or even more frames into the future depending on how many missing
        frames there are inside the file container. This is likely some combination of
        file corruption and OpenCV bug.

        Args:
            index: The frame's index.
        """
        times = self._frameTimes()
        start = index
        back = 1
        while(True):
            self.capture.set(cv2.CAP_PROP_POS_MSEC, times[start])
            if(self.capture.grab()):
                time: float = self.capture.get(cv2.CAP_PROP_POS_MSEC)
                landed: int = bisect.bisect_left(times, time)
                if(landed <= index):
                    self.nextPosition = landed + 1
                    return
            if(start == 0):
                break
            start = max(0, index - back)
            back *= 2
        # We gave up on rewinding, so just close the capture. Callers will start searching from frame 0.
        self.close()

    @override
    def readStored(self, index: int) -> np.ndarray | None:
        # Skip seeking when the next frame is already what you want.
        # This improves performance drastically.
        shouldSeek = index != self.nextPosition
        if(shouldSeek):
            # Don't seek to some far off place beyond the end of the frame.
            if(index >= len(self._frameTimes())):
                return None
            self._seekBefore(index)
            if(self.nextPosition == index + 1):
                return self._retrieve()
        # Read forward to the desired frame.
        while(self.capture.grab()):
            currentPosition = self.nextPosition
            self.nextPosition += 1
            if(currentPosition == index):
                return self._retrieve()
        self.nextPosition = None
        return None

    def _retrieve(self) -> np.ndarray | None:
        """Decodes the frame the capture last grabbed.

        Returns:
            The frame, or None if it can't be decoded.
        """
        success, image = self.capture.retrieve()
        if(not success):
            self.nextPosition = None
            return None
        return image

    @override
    def frameSize(self) -> tuple[int, int]:
        return (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @override
    def close(self) -> None:
        super().close()
        if(self._capture is not None):
            self._capture.release()
            self._capture = None
            self.nextPosition = 0
