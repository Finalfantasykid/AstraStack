from __future__ import annotations
from lazy import lazy
cv2 = lazy("cv2")
av = lazy("av")
np = lazy("numpy")
rawpy = lazy("rawpy")
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
    """A video file decoded with PyAV.

    Frames are indexed by their position in presentation order and found by seeking to their
    timestamps. Reading a frame less than a second after the last one read decodes forward
    instead of seeking. The timestamps are read from the video's packets, without decoding,
    the first time they are needed, and pickled so worker processes don't read them again.
    """

    def __init__(self, path: str) -> None:
        """Initializes the source.

        Args:
            path: The video file.
        """
        self.path = path
        self.timestamps: list[int] | None = None
        self.container: av.container.InputContainer | None = None

    def __getstate__(self) -> dict[str, str | list[float] | None]:
        """Only pickle things that are relevant to a new process.

        E.g. force new processes to open new file descriptors by *not* pickling self._capture.
        """
        return {"path": self.path, "timestamps": self.timestamps}

    def __setstate__(self, state: dict[str, str | list[int] | None]) -> None:
        """Restores a pickled source, which opens its container when first read.

        Args:
            state: The state from __getstate__.
        """
        self.path = state["path"]
        self.timestamps = state["timestamps"]
        self.container = None

    @staticmethod
    def canOpen(path: str | list[str]) -> bool:
        """Accepts a single file PyAV demuxes at least two video frames from.

        One-frame videos are rejected, and left to other FrameSource implementations to handle.
        """
        if(not isinstance(path, str)):
            return False
        try:
            with av.open(path) as container:
                stream = container.streams.video[0]
                packets = 0
                for packet in container.demux(stream):
                    packets += packet.size > 0
                    if(packets >= 2):
                        return True
        except Exception:
            pass
        return False

    def filePaths(self) -> list[str]:
        return [self.path]
        
    def openContainer(self) -> None:
        """Opens the video, if needed, and describes it from its stream's color tags."""
        if(self.container is not None):
            return
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        codec = self.stream.codec_context
        self.width, self.height = codec.width, codec.height
        self.decoder = None
        self.lastTimestamp = None
        self.lastFrame = None

    def frameTimestamps(self) -> list[int]:
        """Returns the presentation timestamps of the frames, read from the packets the first time.

        Returns:
            The sorted timestamps, in the stream's time base.
        """
        if(self.timestamps is None):
            self.openContainer()
            timestamps = set()
            for packet in self.container.demux(self.stream):
                timestamp = packet.pts if packet.pts is not None else packet.dts
                if(timestamp is not None and packet.size > 0):
                    timestamps.add(timestamp)
            self.decoder = None
            self.timestamps = sorted(timestamps)
        return self.timestamps

    def frameCount(self) -> int:
        """The count is the number of distinct timestamps among the video's packets."""
        return len(self.frameTimestamps())

    def frameToImage(self, videoFrame: av.VideoFrame) -> np.ndarray:
        image = None
        if("bayer_" in videoFrame.format.name):
            # Return a Greyscale mozaiced image
            # TODO: This should also inform the bayer guess
            image = videoFrame.to_ndarray()
        elif("yuv" in videoFrame.format.name or videoFrame.format.is_rgb):
            # Return an BGR image
            image = cv2.cvtColor(videoFrame.to_rgb().to_ndarray(), cv2.COLOR_RGB2BGR)
        else:
            # Return a Greyscale image
            image = videoFrame.to_ndarray()
        return image

    def readFrame(self, index: int) -> np.ndarray | None:
        """Frames are converted to 16-bit BGR, whatever the stream's depth."""
        if(index == 0 and self.timestamps is None):
            # Decoded in a container of its own, without reading every packet's timestamp, so thumbnails are quick
            self.openContainer()
            with av.open(self.path) as container:
                videoFrame = next(container.decode(video=0), None)
            if(videoFrame is None):
                return None
            return self.frameToImage(videoFrame)
        timestamps = self.frameTimestamps()
        if(index >= len(timestamps)):
            return None
        self.openContainer()
        target = timestamps[index]
        if(self.lastFrame is None or target != self.lastTimestamp):
            if(self.decoder is None or self.lastTimestamp is None or target < self.lastTimestamp or
               target - self.lastTimestamp > 1/self.stream.time_base):
                self.container.seek(target, stream=self.stream, backward=True)
                self.decoder = self.container.decode(self.stream)
            self.lastFrame = None
            for videoFrame in self.decoder:
                timestamp = videoFrame.pts if videoFrame.pts is not None else videoFrame.dts
                if(timestamp is not None and timestamp >= target):
                    self.lastTimestamp = timestamp
                    self.lastFrame = videoFrame
                    break
            if(self.lastFrame is None):
                self.decoder = None
                return None
        if(self.lastTimestamp != target):
            # The frame couldn't be decoded
            return None
        return self.frameToImage(self.lastFrame)

    def frameSize(self) -> tuple[int, int]:
        self.openContainer()
        return (self.width, self.height)

    def close(self) -> None:
        if(self.container is not None):
            self.container.close()
            self.container = None
