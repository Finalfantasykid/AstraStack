import cv2
import glob
from math import ceil
from Globals import g

class Video:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.frames = []
        self.sharpest = 0
        self.vidcap = None
            
    # Checks to see if there will be enough memory to process the image
    def checkMemory(self):
        if(isinstance(g.file, list)):
            # Image Sequence
            for file in sorted(g.file):
                image = cv2.imread(file)
                h, w = image.shape[:2]
                if(not g.ui.checkMemory(w, h)):
                    raise MemoryError()
                return
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
        def progress(msg):
            self.count += 1
            g.ui.setProgress(self.count, self.total, msg)
        g.ui.createListener(progress)

        sharps = []
        if(isinstance(g.file, list)):
            # Image Sequence
            self.total = len(g.file)
            i = 0
            w = 0
            h = 0
            for file in sorted(g.file):
                image = cv2.imread(file)
                height, width = image.shape[:2]
                if(w == 0 and h == 0):
                    # Set height, width of first frame
                    h = height
                    w = width
                if(w == width and h == height):
                    # Only add if dimensions match
                    self.frames.append(file)
                    # Calculate sharpness
                    sharps.append(calculateSharpness(image))
                    i += 1
                self.count += 1
                g.ui.setProgress(self.count, self.total, "Loading Frames")
        else:
            # Video
            vidcap = cv2.VideoCapture(g.file)
            self.total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            countPerThread = ceil(self.total/g.nThreads)
            
            futures = []
            for t in range(0, g.nThreads):
                futures.append(g.pool.submit(loadFrames, g.file, t*countPerThread, countPerThread, g.ui.childConn))
            for t in range(0, g.nThreads):
                frames, sharp = futures[t].result()
                self.frames += frames
                sharps += sharp
            vidcap.release()
            
        # Some videos are weird with their frame timings, so get rid of possible duplicates
        framesDict = dict()
        tmpFrames = []
        tmpSharps = []
        for i, frame in enumerate(self.frames):
            if(frame not in framesDict):
                framesDict[frame] = True
                tmpFrames.append(frame)
                tmpSharps.append(sharps[i])
        
        sharps = tmpSharps
        self.frames = tmpFrames
        self.total = len(self.frames)

        if(len(sharps) > 0):
            self.sharpest = sharps.index(max(sharps))
        else:
            self.sharpest = 0

        g.ui.finishedVideo()
        g.ui.childConn.send("stop")
        
    def getFrame(self, file, frame):
        if(isinstance(frame, str)):
            # Specific file
             image = cv2.imread(frame)
        elif(isinstance(file, list)):
            # Image Sequence
            files = sorted(file)
            image = cv2.imread(files[frame])
        else:
            # Video
            if(self.vidcap is None):
                self.vidcap = cv2.VideoCapture(file)
            frameTime = ((1000/self.vidcap.get(cv2.CAP_PROP_FPS))+0.1)
            lastTime = self.vidcap.get(cv2.CAP_PROP_POS_MSEC)
            if(not (frame < lastTime + frameTime and frame > lastTime)):
                self.vidcap.set(cv2.CAP_PROP_POS_MSEC, frame)
            success,image = self.vidcap.read()
        return image
       
# Returns a number based on how sharp the image is (higher = sharper) 
def calculateSharpness(image):
    h, w = image.shape[:2]
    return cv2.Laplacian(cv2.resize(image, (int(max(100, w*0.1)), int(max(100, h*0.1)))), cv2.CV_8U).var()
        
# Multiprocess function to load frames
def loadFrames(file, start, count, conn):
    vidcap = cv2.VideoCapture(file)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    sharps = []
    for i in range(0, count):
        success,image = vidcap.read()
        if(success):
            conn.send("Loading Frames")
            frames.append(vidcap.get(cv2.CAP_PROP_POS_MSEC))
            # Calculate sharpness
            sharps.append(calculateSharpness(image))
    vidcap.release()
    return (frames, sharps)
