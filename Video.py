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

    # Returns a list of file paths for the frames of the given video fileName
    def run(self):
        def progress(msg):
            self.count += 1
            g.ui.setProgress(self.count, self.total, msg)
        g.ui.createListener(progress)
        
        if not path.exists(g.tmp + "frames"):
            makedirs(g.tmp + "frames")
        if not path.exists(g.tmp + "cache"):
            makedirs(g.tmp + "cache")
        vidcap = cv2.VideoCapture(g.file)
        self.total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        countPerThread = ceil(self.total/g.nThreads)
        
        futures = []
        for t in range(0, g.nThreads):
            futures.append(g.pool.submit(loadFrames, g.file, t*countPerThread, countPerThread))
        for t in range(0, g.nThreads):
            self.frames += futures[t].result()

        g.ui.setProgress()
        g.ui.finishedVideo()
        g.ui.childConn.send("stop")
        
# Multiprocess function to load frames
def loadFrames(file, start, count):
    vidcap = cv2.VideoCapture(file)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for i in range(0, count):
        success,image = vidcap.read()
        if(success):
            fileName = g.tmp + "frames/%d.png" % (start + i)
            cv2.imwrite(fileName, image)
            g.ui.childConn.send("Loading Frames")
            frames.append(fileName)
    return frames
