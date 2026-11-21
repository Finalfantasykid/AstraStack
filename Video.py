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
        toDtype: type = np.float32,
        fastestDemosaic: bool = False,
    ) -> np.ndarray:
        """Reads a frame of the source as BGR, demosaiced as the source's demosaic params say.

        Args:
            index: The frame's index in the source.
            toDtype: The output's dtype, as for FrameSource.convertDepth.
            fastestDemosaic: Whether to demosaic with the fastest algorithm, as for FrameSource.readFrame.

        Returns:
            The frame as BGR.
        """
        return FrameSource.convertDepth(
            image=self.source.readFrame(index, fastestDemosaic=fastestDemosaic),
            toDtype=toDtype,
        )

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
               progress: ProgressCounter) -> tuple[list[int], list[float], list[np.ndarray]]:
    """Reads a chunk of frames, calculating each one's sharpness and its delta from the previous frame.

    Runs in a worker process.

    Args:
        source: The source to read from, demosaiced as its demosaic params say before calculating sharpness and
            deltas.
        indices: The consecutive frame indices of the chunk. The frame before the chunk is read
            too, as the previous frame for the first delta.
        frameSize: The (width, height) frames must have; frames of any other size are skipped.
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
                FrameSource.convertDepth(
                    image=prevImage,
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
        image = FrameSource.convertDepth(
            image=image,
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
            Video(source).getFrame(index=0),
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
