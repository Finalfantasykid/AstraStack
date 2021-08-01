import cv2
import numpy as np
import glob
import math
from concurrent.futures.process import BrokenProcessPool
from natsort import natsorted, ns
from Globals import g
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
        progress = ProgressBar()
        sharps = []
        futures = []
        try:
            if(isinstance(g.file, list)):
                # Image Sequence
                progress.total = len(g.file)
                countPerThread = math.ceil(progress.total/g.nThreads)
                g.file = natsorted(g.file, alg=ns.IGNORECASE)
                
                g.guessedColorMode = self.guessColorMode(g.file)
                
                progress.setMessage("Loading Frames")
                height, width = cv2.imread(g.file[0]).shape[:2]
                for i in range(0, g.nThreads):
                    futures.append(g.pool.submit(loadFramesSequence, g.file[i*countPerThread:(i+1)*countPerThread], width, height, (g.colorMode or g.guessedColorMode), ProgressCounter(progress.counter(i), g.nThreads)))
                for i in range(0, g.nThreads):
                    frames, sharp = futures[i].result()
                    self.frames += frames
                    sharps += sharp
            else:
                # Video
                self.vidcap = cv2.VideoCapture(g.file)
                progress.total = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

                g.guessedColorMode = self.guessColorMode(g.file)
                
                progress.setMessage("Loading Frames")
                countPerThread = math.ceil(progress.total/g.nThreads)
                for i in range(0, g.nThreads):
                    futures.append(g.pool.submit(loadFramesVideo, g.file, i*countPerThread, countPerThread, (g.colorMode or g.guessedColorMode), ProgressCounter(progress.counter(i), g.nThreads)))
                for i in range(0, g.nThreads):
                    frames, sharp = futures[i].result()
                    self.frames += frames
                    sharps += sharp
        except BrokenProcessPool:
            progress.stop()
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

        if(len(sharps) > 0):
            self.sharpest = sharps.index(max(sharps))
        else:
            self.sharpest = 0

        g.ui.finishedVideo()
        progress.stop()
        
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
            if(diff > 1):
                # Uses a bayer pattern, but which one...
                rggbDiff = np.average(cv2.absdiff(rggb[:,:,0], rggb[:,:,2]))
                grbgDiff = np.average(cv2.absdiff(grbg[:,:,0], grbg[:,:,2]))
                if(rggbDiff >= grbgDiff):
                    # Probably either RGGB or BGGR
                    colorMode = Video.COLOR_RGGB
                else:
                    # Probably either GRBG or GBRG
                    colorMode = Video.COLOR_GRBG
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
       
# Returns a number based on how sharp the image is (higher = sharper)
def calculateSharpness(image):
    h, w = image.shape[:2]
    return cv2.Laplacian(cv2.resize(image, (int(max(100, w*0.1)), int(max(100, h*0.1)))), cv2.CV_8U).var()
   
# Multiprocess function to load frames from an image sequence
def loadFramesSequence(files, width, height, colorMode, progress):
    frames = []
    sharps = []
    for i, file in enumerate(files):
        progress.count(i, len(files))
        image = cv2.imread(file)
        h, w = image.shape[:2]
        if(h == height and w == width):
            # Only add if dimensions match
            frames.append(file)
            # Calculate sharpness
            image = Video.colorMode(image, colorMode)
            sharps.append(calculateSharpness(image))
    progress.countExtra()
    return (frames, sharps)
        
# Multiprocess function to load frames from a video source
def loadFramesVideo(file, start, count, colorMode, progress):
    vidcap = cv2.VideoCapture(file)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    sharps = []
    for i in range(0, count):
        success,image = vidcap.read()
        if(success):
            progress.count(i, count)
            frames.append(vidcap.get(cv2.CAP_PROP_POS_MSEC))
            # Calculate sharpness
            image = Video.colorMode(image, colorMode)
            sharps.append(calculateSharpness(image))
    progress.countExtra()
    vidcap.release()
    return (frames, sharps)
