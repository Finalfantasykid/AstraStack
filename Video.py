from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
import math
from concurrent.futures.process import BrokenProcessPool
from Globals import *
from WorkerPools import FRAME_POOL
from ProgressBar import *
from FrameSource import FrameSource, ImageSequenceSource

class Video:

    COLOR_AUTO = 0
    COLOR_RGB = 1
    COLOR_GRAYSCALE = 2
    COLOR_RGGB = 3
    COLOR_GRBG = 4
    COLOR_GBRG = 5
    COLOR_BGGR = 6
    COLOR_RGGB_VNG = 7
    COLOR_GRBG_VNG = 8
    COLOR_GBRG_VNG = 9
    COLOR_BGGR_VNG = 10

    def __init__(self, source: FrameSource) -> None:
        self.source = source
        # Source indices of the frames kept on load; wrong-size or unreadable frames are left out
        self.frames: list[int] = []
        self.sharpest = 0
        self.sharps = []
        self.deltas = []
        
    # Checks to see if there will be enough memory to process the image
    def checkMemory(self) -> None:
        width, height = self.source.frameSize()
        if(not g.ui.checkMemory(w=width, h=height)):
            raise MemoryError()

    # Loads the frames of the source, calculating their sharpness and frame to frame deltas
    def run(self) -> None:
        progress = ProgressBar()
        futures = []
        try:
            progress.total = self.source.frameCount()
            countPerThread = math.ceil(progress.total/g.nThreads)

            g.guessedColorMode = self.guessColorMode()

            progress.setMessage("Preprocessing Frames")
            frameSize = self.source.frameSize()
            # Release the main process's file handles while the workers read; each worker opens its own
            self.source.close()
            for i in range(0, g.nThreads):
                indices = range(i*countPerThread, (i+1)*countPerThread)
                futures.append(FRAME_POOL.submit(loadFrames,
                                                 source=self.source,
                                                 indices=indices,
                                                 frameSize=frameSize,
                                                 colorMode=g.actualColor(),
                                                 progress=ProgressCounter(progress.counter(i), g.nThreads)))
            for i in range(0, g.nThreads):
                frames, sharps, deltas = futures[i].result()
                self.frames += frames
                self.sharps += sharps
                self.deltas += deltas
        except BrokenProcessPool:
            progress.stop()
            return

        if(len(self.sharps) > 0):
            self.sharpest = self.sharps.index(max(self.sharps))
        else:
            self.sharpest = 0

        progress.stop()
        g.ui.finishedVideo()

    def skippedCount(self) -> int:
        """Returns how many of the source's frames were left out on load."""
        return self.source.frameCount() - len(self.frames)

    def getFrame(
        self,
        index: int,
        colorMode: int,
        toDtype: type = np.float32,
    ) -> np.ndarray:
        """Reads a frame of the source and converts it to BGR.

        Args:
            index: The frame's index in the source.
            colorMode: The color mode the frame is in; COLOR_AUTO resolves it from the source
                and the frame.
            toDtype: The output's dtype, as for _bgrFromSourceFrame.

        Returns:
            The frame as BGR.
        """
        image = self.source.readFrame(index)
        if colorMode == Video.COLOR_AUTO:
            colorMode = self.guessColorMode(image)
        return Video._bgrFromSourceFrame(
            image=image,
            colorMode=colorMode,
            toDtype=toDtype,
        )

    def thumbnail(self) -> np.ndarray:
        """Makes a display-ready thumbnail of the source.

        Returns:
            The source's fast preview if it has one, or else its first frame in the Auto color
            mode; as uint8 BGR.
        """
        preview = self.source.fastPreview()
        if preview is not None:
            return preview
        return self.getFrame(
            index=0,
            colorMode=Video.COLOR_AUTO,
            toDtype=np.uint8,
        )

    def guessColorMode(self, image: np.ndarray | None = None) -> int:
        """Resolves the Auto color mode for the source.

        Args:
            image: A frame of the source to guess from; the first frame is read if None.

        Returns:
            The color mode guessed from the frame's pixels.
        """
        if(image is None):
            image = self.source.readFrame(0)
        if(image is None):
            return Video.COLOR_RGB
        return self.guessImageColor(image)

    def guessImageColor(self, image: np.ndarray) -> int:
        """Guesses a frame's color mode from its pixels.

        A very simple guess, far from perfect: a frame with equal channels is grayscale unless
        demosaicing it reveals a Bayer pattern. The pattern is only told apart as RGGB/BGGR or
        GRBG/GBRG, and the user's last chosen Bayer mode replaces it if there is one.

        Args:
            image: A frame as FrameSource.readFrame returns it.

        Returns:
            The guessed color mode.
        """
        # Assume its color to start with
        colorMode = Video.COLOR_RGB
        rgb = Video._bgrFromSourceFrame(image, Video.COLOR_RGB)
        if(np.array_equal(rgb[:,:,0],rgb[:,:,1]) and 
           np.array_equal(rgb[:,:,1],rgb[:,:,2])):
            # Not a color image
            colorMode = Video.COLOR_GRAYSCALE
            
            # Calculate all bayer options
            rggb = Video._bgrFromSourceFrame(image, Video.COLOR_RGGB)
            grbg = Video._bgrFromSourceFrame(image, Video.COLOR_GRBG)
            gbrg = Video._bgrFromSourceFrame(image, Video.COLOR_GBRG)
            bggr = Video._bgrFromSourceFrame(image, Video.COLOR_BGGR)
            
            diff1 = np.average(cv2.absdiff(cv2.cvtColor(rggb, cv2.COLOR_BGR2GRAY), cv2.cvtColor(bggr, cv2.COLOR_BGR2GRAY)))
            diff2 = np.average(cv2.absdiff(cv2.cvtColor(grbg, cv2.COLOR_BGR2GRAY), cv2.cvtColor(gbrg, cv2.COLOR_BGR2GRAY)))
            diff = 0
            if(diff1 > 0.0001 and diff2 > 0.0001):
                diff = abs(diff1 - diff2)/min(diff1, diff2)
            if(diff > 0.2):
                # Probably uses a bayer pattern, but which one...
                rggbDiff = np.average(cv2.absdiff(rggb[:,:,0], rggb[:,:,2]))
                grbgDiff = np.average(cv2.absdiff(grbg[:,:,0], grbg[:,:,2]))
                if(rggbDiff >= grbgDiff):
                    # Probably either RGGB or BGGR
                    colorMode = Video.COLOR_RGGB
                else:
                    # Probably either GRBG or GBRG
                    colorMode = Video.COLOR_GRBG
                colorMode = g.ui.preferences.get("bayerGuess", colorMode)
        return colorMode
      
    @staticmethod
    def _bgrFromSourceFrame(
        image: np.ndarray,
        colorMode: int,
        toDtype: type | None = None,
    ) -> np.ndarray:
        """Converts a frame read from a source to BGR, interpreting it by its color mode.

        The color mode is applied at the frame's depth, then the depth is converted.

        Args:
            image: A frame as FrameSource.readFrame returns it.
            colorMode: The color mode the frame is in. COLOR_AUTO isn't supported; resolve it
                with Video.guessColorMode first.
            toDtype: The output's dtype: np.float32 for a 0-1 scale, np.uint8 for 8 bits
                (16-bit frames keep their high byte), or None to keep the input's dtype.

        Returns:
            The frame as BGR.

        Raises:
            ValueError: If colorMode is COLOR_AUTO, or the frame's dtype or toDtype isn't supported.
        """
        if colorMode == Video.COLOR_AUTO:
            raise ValueError("COLOR_AUTO must be resolved before converting a frame")
        if image.dtype not in (np.uint8, np.uint16):
            raise ValueError(f"Unsupported frame dtype: {image.dtype}")
        if toDtype not in (None, np.float32, np.uint8):
            raise ValueError(f"Unsupported toDtype: {toDtype}")
        image = Video._applyColorMode(img=image, colorMode=colorMode)
        if toDtype == np.float32:
            return image.astype(np.float32) / np.iinfo(image.dtype).max
        if toDtype == np.uint8 and image.dtype == np.uint16:
            return (image >> 8).astype(np.uint8)
        return image

    @staticmethod
    def _applyColorMode(img: np.ndarray, colorMode: int) -> np.ndarray:
        """Demosaics or grays a frame by its color mode, keeping its depth.

        VNG demosaicing is only done on uint8 frames; deeper frames are demosaiced edge-aware.

        Args:
            img: A frame as FrameSource.readFrame returns it.
            colorMode: The color mode the frame is in; not COLOR_AUTO.

        Returns:
            The frame as BGR.
        """
        h, w = img.shape[:2]
        if(len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if(colorMode == Video.COLOR_RGB):
            pass
        elif(colorMode == Video.COLOR_GRAYSCALE):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if(colorMode >= Video.COLOR_RGGB_VNG and img.dtype == np.uint8): # Only do VNG if its a uint8 image
                img = cv2.copyMakeBorder(img, 4, 2, 2, 2, cv2.BORDER_REPLICATE)
                if(colorMode == Video.COLOR_RGGB_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR_VNG)
                elif(colorMode == Video.COLOR_GRBG_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR_VNG)
                elif(colorMode == Video.COLOR_GBRG_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2BGR_VNG)
                elif(colorMode == Video.COLOR_BGGR_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2BGR_VNG)
                img = img[2:h+2,2:w+2]
            else:
                img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
                if(colorMode == Video.COLOR_RGGB or colorMode == Video.COLOR_RGGB_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR_EA)
                elif(colorMode == Video.COLOR_GRBG or colorMode == Video.COLOR_GRBG_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR_EA)
                elif(colorMode == Video.COLOR_GBRG or colorMode == Video.COLOR_GBRG_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2BGR_EA)
                elif(colorMode == Video.COLOR_BGGR or colorMode == Video.COLOR_BGGR_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2BGR_EA)
                img = img[2:h+2,2:w+2]
        return img
      
# Resize image so it does things faster
def resize(image):
    from Align import dilateImg
    image = dilateImg(image, 9, 3)
    h, w = image.shape[:2]
    hScaleFactor = min(1.0, (100/h))
    wScaleFactor = (int(w*hScaleFactor))/w
    image = cv2.resize(image, (int(w*hScaleFactor), int(h*hScaleFactor)), interpolation=cv2.INTER_LINEAR)
    return (image, wScaleFactor, hScaleFactor)
    
# Calculates frame to frame an approximate translation matrix
def calculateDelta(image, prevImage, wScaleFactor, hScaleFactor):
    if(prevImage is not None):
        try:
            T = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,  1e-10)
            (cc, T) = cv2.findTransformECC(image, prevImage, T, cv2.MOTION_TRANSLATION, criteria)
            
            M = np.identity(3, dtype=np.float64)
            M[0][2] = T[0][2]/wScaleFactor # X
            M[1][2] = T[1][2]/hScaleFactor # Y
        except Exception:
            M = np.identity(3)
    else:
        M = np.identity(3)
    return M
       
# Returns a number based on how sharp the image is (higher = sharper)
def calculateSharpness(image):
    h, w = image.shape[:2]
    return cv2.Laplacian(cv2.resize(image, (int(max(100, w*0.1)), int(max(100, h*0.1)))), cv2.CV_8U).var()
   
def loadFrames(source: FrameSource,
               indices: range,
               frameSize: tuple[int, int],
               colorMode: int,
               progress: ProgressCounter) -> tuple[list[int], list[float], list[np.ndarray]]:
    """Reads a chunk of frames, calculating each one's sharpness and its delta from the previous frame.

    Runs in a worker process.

    Args:
        source: The source to read from.
        indices: The consecutive frame indices of the chunk. The frame before the chunk is read
            too, as the previous frame for the first delta.
        frameSize: The (width, height) frames must have; frames of any other size are skipped.
        colorMode: The color mode to apply before calculating sharpness and deltas.
        progress: Counts the frames as they are read.

    Returns:
        The indices of the loaded frames, their sharpness and their deltas.
    """
    frames = []
    sharps = []
    deltas = []

    prevImage = None
    if(indices.start > 0 and len(indices) > 0):
        prevImage = source.readFrame(indices.start - 1)
        if(prevImage is not None):
            prevImage = cv2.cvtColor(
                Video._bgrFromSourceFrame(
                    image=prevImage,
                    colorMode=colorMode,
                    toDtype=np.uint8,
                ),
                cv2.COLOR_BGR2GRAY,
            )
            prevImage, wScaleFactor, hScaleFactor = resize(prevImage)

    for i, index in enumerate(indices):
        progress.count(i, len(indices))
        image = source.readFrame(index)
        if(image is None):
            continue
        h, w = image.shape[:2]
        if((w, h) != frameSize):
            # Only add if dimensions match
            continue
        frames.append(index)
        # Sharpness and deltas are measured at 8 bits, which their thresholds were tuned on
        image = Video._bgrFromSourceFrame(
            image=image,
            colorMode=colorMode,
            toDtype=np.uint8,
        )

        # Calculate sharpness
        sharps.append(calculateSharpness(image))

        # Calculate Frame Deltas
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image, wScaleFactor, hScaleFactor = resize(image)
        M = calculateDelta(image=image,
                           prevImage=prevImage,
                           wScaleFactor=wScaleFactor,
                           hScaleFactor=hScaleFactor)
        deltas.append(M)

        prevImage = image
    progress.countExtra()
    return (frames, sharps, deltas)


def loadPsf(path: str) -> np.ndarray:
    """Reads an image file as a custom PSF.

    Args:
        path: The image file.

    Returns:
        The PSF as grayscale float32, stretched to 0-1.

    Raises:
        ValueError: If the image is larger than 100x100 or has only one value.
    """
    source = ImageSequenceSource(path)
    try:
        psf = cv2.cvtColor(
            Video(source).getFrame(index=0, colorMode=Video.COLOR_GRAYSCALE),
            cv2.COLOR_BGR2GRAY,
        )
    finally:
        source.close()
    if max(psf.shape) > 100:
        raise ValueError("The PSF must be no larger than 100x100")
    if len(np.unique(psf)) <= 1:
        raise ValueError("The PSF must have more than one value")
    cv2.normalize(psf, psf, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return psf
