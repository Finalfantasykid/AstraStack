import cv2
import glob
from os import path, makedirs, scandir, unlink
from Globals import g

class Video:

    def __init__(self):
        self.frames = []

    # Returns a list of file paths for the frames of the given video fileName
    def run(self):
        if not path.exists("frames"):
            makedirs("frames")
        if not path.exists("cache"):
            makedirs("cache")
        self.cleanFrames()
        vidcap = cv2.VideoCapture(g.file)
        count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        success,image = vidcap.read()
        
        i = 0
        success = True
        while success:
            g.ui.setProgress(i, count-1, "Loading Frames")
            fileName = "frames/%d.png" % i
            cv2.imwrite(fileName, image)
            success,image = vidcap.read()
            self.frames.append(fileName)
            i += 1
        g.ui.setProgress()
        g.ui.finishedVideo()
    
    def cleanFrames(self):
        for file in scandir("frames"):
            unlink(file.path)
        for file in scandir("cache"):
            unlink(file.path)
