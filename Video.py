import cv2
import numpy as np
import glob
import math
from concurrent.futures.process import BrokenProcessPool
from natsort import natsorted, ns
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
        def progress(msg):
            self.count += 1
            g.ui.setProgress(self.count, self.total, msg)
        g.ui.createListener(progress)

        sharps = []
        futures = []
        try:
            if(isinstance(g.file, list)):
                # Image Sequence
                self.total = len(g.file)
                countPerThread = math.ceil(self.total/g.nThreads)
                g.file = natsorted(g.file, alg=ns.IGNORECASE)
                height, width = cv2.imread(g.file[0]).shape[:2]
                
                for i in range(0, g.nThreads):
                    futures.append(g.pool.submit(loadFramesSequence, g.file[i*countPerThread:(i+1)*countPerThread], width, height, g.ui.childConn))
                for i in range(0, g.nThreads):
                    frames, sharp = futures[i].result()
                    self.frames += frames
                    sharps += sharp
            else:
                # Video
                vidcap = cv2.VideoCapture(g.file)
                self.total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                countPerThread = math.ceil(self.total/g.nThreads)

                for i in range(0, g.nThreads):
                    futures.append(g.pool.submit(loadFramesVideo, g.file, i*countPerThread, countPerThread, g.ui.childConn))
                for i in range(0, g.nThreads):
                    frames, sharp = futures[i].result()
                    self.frames += frames
                    sharps += sharp
                vidcap.release()
        except BrokenProcessPool:
            g.ui.childConn.send("stop")
            return
            
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
            image = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
            image = (image.astype(np.float32)/np.iinfo(image.dtype).max)*255
        elif(isinstance(file, list)):
            # Image Sequence
            image = cv2.imread(file[frame], cv2.IMREAD_UNCHANGED)
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
        return image
       
# Returns a number based on how sharp the image is (higher = sharper)
def calculateSharpness(image):
    h, w = image.shape[:2]
    return cv2.Laplacian(cv2.resize(image, (int(max(100, w*0.1)), int(max(100, h*0.1)))), cv2.CV_8U).var()
   
# Multiprocess function to load frames from an image sequence
def loadFramesSequence(files, width, height, conn):
    frames = []
    sharps = []
    for file in files:
        conn.send("Loading Frames")
        image = cv2.imread(file)
        h, w = image.shape[:2]
        if(h == height and w == width):
            # Only add if dimensions match
            frames.append(file)
            # Calculate sharpness
            sharps.append(calculateSharpness(image))
    return (frames, sharps)
        
# Multiprocess function to load frames from a video source
def loadFramesVideo(file, start, count, conn):
    vidcap = cv2.VideoCapture(file)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    sharps = []
    for i in range(0, count):
        success,image = vidcap.read()
        conn.send("Loading Frames")
        if(success):
            frames.append(vidcap.get(cv2.CAP_PROP_POS_MSEC))
            # Calculate sharpness
            sharps.append(calculateSharpness(image))
    vidcap.release()
    return (frames, sharps)
