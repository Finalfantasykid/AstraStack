import cv2
import glob
from math import ceil
from os import path, makedirs
from Globals import g

class Video:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.frames = []
        self.sharpest = 0

    def mkdirs(self):
        if not path.exists(g.tmp + "frames"):
            makedirs(g.tmp + "frames")
        if not path.exists(g.tmp + "cache"):
            makedirs(g.tmp + "cache")
            
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
        
        self.mkdirs()
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
                    fileName = g.tmp + "frames/%d.png" % i
                    cv2.imwrite(fileName, image)
                    self.frames.append(fileName)
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

        if(len(sharps) > 0):
            self.sharpest = sharps.index(max(sharps))
        else:
            self.sharpest = 0

        g.ui.setProgress()
        g.ui.finishedVideo()
        g.ui.childConn.send("stop")
       
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
            fileName = g.tmp + "frames/%d.png" % (start + i)
            cv2.imwrite(fileName, image)
            conn.send("Loading Frames")
            frames.append(fileName)
            # Calculate sharpness
            sharps.append(calculateSharpness(image))
    return (frames, sharps)
