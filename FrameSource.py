from __future__ import annotations
from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
rawpy = lazy("rawpy")
import bisect
from abc import ABC, abstractmethod
from natsort import natsorted, ns

class FrameSource(ABC):
    """A sequence of frames stored in one or more files, addressed by integer index.

    Sources are sent to worker processes, so they must pickle as just what identifies their
    frames and open files lazily wherever they are read.
    """

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
        for source in (ImageSequenceSource, VideoSource):
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
    def readFrame(self, index: int) -> np.ndarray | None:
        """Reads one frame as stored.

        Args:
            index: The frame's index.

        Returns:
            The frame in BGR order, or with a single channel for gray or Bayer mosaic frames,
            at its stored depth (uint8 or uint16); or None if the source has no readable frame
            at index.
        """

    @abstractmethod
    def frameSize(self) -> tuple[int, int]:
        """Returns the size of the frames.

        Returns:
            The (width, height) of the first frame.
        """

    def fastPreview(self) -> np.ndarray | None:
        """Reads a display-ready preview of the first frame, if the file offers a quick way to.

        Return None unless this is much quicker than reading the first frame; the caller then
        makes the preview from readFrame(0) itself.

        Returns:
            The preview as uint8 BGR, possibly at a reduced size; or None.
        """
        return None

    def close(self) -> None:
        """Releases any open file handles. The source reopens them if it is read again."""

class ImageSequenceSource(FrameSource):
    """A list of image files, one frame per file."""

    def __init__(self, path: str | list[str]) -> None:
        """Initializes the source.

        Args:
            path: An image file, or a list of image files (in any order).
        """
        self.paths = [path] if isinstance(path, str) else natsorted(path, alg=ns.IGNORECASE)

    @staticmethod
    def canOpen(path: str | list[str]) -> bool:
        """Accepts an image file, or a list of one or more image files.

        Each file must be recognized, without decoding its pixels, by one of the readers
        readFrame decodes with, in its order: a raw camera file, an image format OpenCV reads,
        then a FITS file with image data.
        """
        paths = [path] if isinstance(path, str) else path
        for file in paths:
            try:
                with rawpy.imread(file):
                    continue
            except Exception:
                pass
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

    def filePaths(self) -> list[str]:
        return self.paths

    def frameCount(self) -> int:
        return len(self.paths)

    def readFrame(self, index: int) -> np.ndarray | None:
        """Raw camera files are decoded and FITS files scaled to 16 bits."""
        if(index >= len(self.paths)):
            return None
        return ImageSequenceSource._readImageFile(self.paths[index])

    @staticmethod
    def _readImageFile(file: str) -> np.ndarray:
        """Decodes an image file as a raw camera file, else an image OpenCV reads, else FITS.

        Args:
            file: The image file.

        Returns:
            The image. A raw camera image is demosaiced to 16-bit BGR, an image OpenCV reads
            keeps its stored depth and channels, and a FITS image is stretched to the full
            16-bit range.
        """
        try:
            raw = rawpy.imread(file)
        except Exception:
            raw = None
        if raw is not None:
            with raw:
                return cv2.cvtColor(raw.postprocess(output_bps=16), cv2.COLOR_RGB2BGR)
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

    def fastPreview(self) -> np.ndarray | None:
        """A raw camera file is decoded at half size and 8 bits; other files have none."""
        # TODO: Read a raw camera file's embedded JPEG preview (rawpy's extract_thumb) instead,
        # falling back to the half size decode for files without one. It is far quicker, and it
        # keeps LibRaw's OpenMP code out of the main process.
        try:
            raw = rawpy.imread(self.paths[0])
        except Exception:
            return None
        with raw:
            return cv2.cvtColor(
                raw.postprocess(half_size=True, output_bps=8),
                cv2.COLOR_RGB2BGR,
            )

    def frameSize(self) -> tuple[int, int]:
        height, width = self.readFrame(0).shape[:2]
        return (width, height)

class VideoSource(FrameSource):
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

    def __getstate__(self) -> dict[str, str | list[float] | None]:
        """Only pickle things that are relevant to a new process.

        E.g. force new processes to open new file descriptors by *not* pickling self._capture.
        """
        return {"path": self.path, "times": self.times}

    def __setstate__(self, state: dict[str, str | list[float] | None]) -> None:
        """Restores a pickled source, which opens its capture when first read."""
        self.path = state["path"]
        self.times = state["times"]
        self._capture = None
        self.nextPosition = 0

    @staticmethod
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

    def readFrame(self, index: int) -> np.ndarray | None:
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

    def frameSize(self) -> tuple[int, int]:
        return (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def close(self) -> None:
        if(self._capture is not None):
            self._capture.release()
            self._capture = None
            self.nextPosition = 0
