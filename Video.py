from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
import math
from concurrent.futures.process import BrokenProcessPool
from Globals import *
from ProgressBar import *

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

    def __init__(self):
        self.frames = []
        self.sharpest = 0
        self.vidcap = None
        self.sharps = []
        self.deltas = []
            
    # Checks to see if there will be enough memory to process the image
    def checkMemory(self):
        if(isinstance(g.file, list)):
            # Image Sequence
            h, w = cv2.imread(g.file[0]).shape[:2]
            if(not g.ui.checkMemory(w, h)):
                raise MemoryError()
        else:
            # Video
            vidcap = cv2.VideoCapture(g.file)
            width  = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            vidcap.release()
            if(not g.ui.checkMemory(w=width,h=height)):
                raise MemoryError()

    # Returns a list of file paths for the frames of the given video fileName
    def run(self):
        from natsort import natsorted, ns
        progress = ProgressBar()
        futures = []
        try:
            if(isinstance(g.file, list)):
                # Image Sequence
                progress.total = len(g.file)
                countPerThread = math.ceil(progress.total/g.nThreads)
                g.file = natsorted(g.file, alg=ns.IGNORECASE)
                
                g.guessedColorMode = self.guessColorMode(g.file)
                
                progress.setMessage("Preprocessing Frames")
                height, width = cv2.imread(g.file[0]).shape[:2]
                for i in range(0, g.nThreads):
                    futures.append(g.pool.submit(loadFramesSequence, g.file[max(0, (i*countPerThread)-1):(i+1)*countPerThread], i, width, height, g.actualColor(), ProgressCounter(progress.counter(i), g.nThreads)))
                for i in range(0, g.nThreads):
                    frames, sharp, deltas = futures[i].result()
                    self.frames += frames
                    self.sharps += sharp
                    self.deltas += deltas
            else:
                # Video
                self.vidcap = cv2.VideoCapture(g.file)
                progress.total = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                frameTime = (1000/self.vidcap.get(cv2.CAP_PROP_FPS))

                g.guessedColorMode = self.guessColorMode(g.file)
                
                progress.setMessage("Preprocessing Frames")
                countPerThread = math.ceil(progress.total/g.nThreads)
                for i in range(0, g.nThreads):
                    futures.append(g.pool.submit(loadFramesVideo, g.file, i*countPerThread, countPerThread, g.actualColor(), ProgressCounter(progress.counter(i), g.nThreads)))
                for i in range(0, g.nThreads):
                    frames, sharp, deltas, wScaleFactor, hScaleFactor = futures[i].result()
                    self.frames += frames
                    self.sharps += sharp
                    self.deltas += deltas
                    
                framesDict = dict()
                tmpFrames = []
                tmpSharps = []
                tmpDeltas = []
                for i, frame in enumerate(self.frames):
                    # Work-around for https://github.com/opencv/opencv/issues/20550
                    if(i > 0 and frame == 0):
                        frame = tmpFrames[-1] + frameTime
                    
                    # Some videos are weird with their frame timings, so get rid of possible duplicates
                    if(frame not in framesDict):
                        framesDict[frame] = True
                        tmpFrames.append(frame)
                        tmpSharps.append(self.sharps[i])
                        tmpDeltas.append(self.deltas[i])
                self.sharps = tmpSharps
                self.frames = tmpFrames
                self.deltas = tmpDeltas
                
        except BrokenProcessPool:
            progress.stop()
            return

        if(len(self.sharps) > 0):
            self.sharpest = self.sharps.index(max(self.sharps))
        else:
            self.sharpest = 0

        progress.stop()
        g.ui.finishedVideo()
        
    def getFrame(self, file, frame, colorMode):
        if(colorMode == Video.COLOR_AUTO):
            colorMode = self.guessColorMode(file)
        if(isinstance(frame, str)):
            # Specific file
            image = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
            image = Video.colorMode(image, colorMode)
            image = (image.astype(np.float32)/np.iinfo(image.dtype).max)*255
        elif(isinstance(file, list)):
            # Image Sequence
            image = cv2.imread(file[frame], cv2.IMREAD_UNCHANGED)
            image = Video.colorMode(image, colorMode)
            image = (image.astype(np.float32)/np.iinfo(image.dtype).max)*255
        else:
            # Video
            if(self.vidcap is None):
                self.vidcap = cv2.VideoCapture(file)
            frameTime = ((1000/self.vidcap.get(cv2.CAP_PROP_FPS))+0.1)
            lastTime = self.vidcap.get(cv2.CAP_PROP_POS_MSEC)
            if(not (frame < lastTime + frameTime and frame > lastTime)):
                self.vidcap.set(cv2.CAP_PROP_POS_MSEC, frame)
            success,image = self.vidcap.read()
            image = Video.colorMode(image, colorMode)
        return image
      
    def guessColorMode(self, file):
        colorMode = Video.COLOR_RGB
        if(isinstance(file, list)):
            # Image Sequence
            image = cv2.imread(file[0], cv2.IMREAD_UNCHANGED)
            colorMode = self.guessImageColor(image)
        else:
            try:
                # First try as image
                image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                colorMode = self.guessImageColor(image)
            except Exception:
                # Now try as video
                vidcap = cv2.VideoCapture(file)
                success,image = vidcap.read()
                colorMode = self.guessImageColor(image)
                vidcap.release()
        return colorMode
      
    # Very simple auto-detect of image type (far from perfect)
    def guessImageColor(self, image):
        # Assume its color to start with
        colorMode = Video.COLOR_RGB
        rgb = Video.colorMode(image, Video.COLOR_RGB)
        if(np.array_equal(rgb[:,:,0],rgb[:,:,1]) and 
           np.array_equal(rgb[:,:,1],rgb[:,:,2])):
            # Not a color image
            colorMode = Video.COLOR_GRAYSCALE
            
            # Calculate all bayer options
            rggb = Video.colorMode(image, Video.COLOR_RGGB)
            grbg = Video.colorMode(image, Video.COLOR_GRBG)
            gbrg = Video.colorMode(image, Video.COLOR_GBRG)
            bggr = Video.colorMode(image, Video.COLOR_BGGR)
            
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
      
    # Changes the color mode of the image
    def colorMode(img, colorMode):
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
            if(colorMode >= Video.COLOR_RGGB_VNG):
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
                if(colorMode == Video.COLOR_RGGB):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR_EA)
                elif(colorMode == Video.COLOR_GRBG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR_EA)
                elif(colorMode == Video.COLOR_GBRG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2BGR_EA)
                elif(colorMode == Video.COLOR_BGGR):
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
        except Exception as e:
            M = np.identity(3)
    else:
        M = np.identity(3)
    return M
       
# Returns a number based on how sharp the image is (higher = sharper)
def calculateSharpness(image):
    h, w = image.shape[:2]
    return cv2.Laplacian(cv2.resize(image, (int(max(100, w*0.1)), int(max(100, h*0.1)))), cv2.CV_8U).var()
   
# Multiprocess function to load frames from an image sequence
def loadFramesSequence(files, thread, width, height, colorMode, progress):
    frames = []
    sharps = []
    deltas = []
    
    prevImage = None
    if(thread > 0 and len(files) > 0):
        prevImage = cv2.imread(files.pop(0))
        prevImage = cv2.cvtColor(Video.colorMode(prevImage, colorMode), cv2.COLOR_BGR2GRAY)
        prevImage, wScaleFactor, hScaleFactor = resize(prevImage)
    
    for i, file in enumerate(files):
        progress.count(i, len(files))
        image = cv2.imread(file)
        h, w = image.shape[:2]
        if(h == height and w == width):
            # Only add if dimensions match
            frames.append(file)
            image = Video.colorMode(image, colorMode)

            # Calculate sharpness
            sharps.append(calculateSharpness(image))
            
            # Calculate Frame Deltas
            image = cv2.cvtColor(Video.colorMode(image, colorMode), cv2.COLOR_BGR2GRAY)
            image, wScaleFactor, hScaleFactor = resize(image)
            M = calculateDelta(image, prevImage, wScaleFactor, hScaleFactor)
            deltas.append(M)
            
            prevImage = image
    progress.countExtra()
    return (frames, sharps, deltas)
        
# Multiprocess function to load frames from a video source
def loadFramesVideo(file, start, count, colorMode, progress):
    beforeStart = max(0, start-1)
    vidcap = cv2.VideoCapture(file)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, beforeStart)
    frames = []
    sharps = []
    deltas = []
    
    prevImage = None
    if(beforeStart < start):
        success,prevImage = vidcap.read()
        prevImage = cv2.cvtColor(Video.colorMode(prevImage, colorMode), cv2.COLOR_BGR2GRAY)
        prevImage, wScaleFactor, hScaleFactor = resize(prevImage)
        
    for i in range(0, count):
        success,image = vidcap.read()
        if(success):
            progress.count(i, count)
            frames.append(vidcap.get(cv2.CAP_PROP_POS_MSEC))
            
            # Calculate sharpness
            sharps.append(calculateSharpness(image))

            # Calculate Frame Deltas
            image = cv2.cvtColor(Video.colorMode(image, colorMode), cv2.COLOR_BGR2GRAY)
            image, wScaleFactor, hScaleFactor = resize(image)
            M = calculateDelta(image, prevImage, wScaleFactor, hScaleFactor)
            deltas.append(M)
            
            prevImage = image
        else:
            progress.count(i, count)
    progress.countExtra()
    vidcap.release()
    return (frames, sharps, deltas, wScaleFactor, hScaleFactor)
