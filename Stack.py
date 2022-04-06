import cv2
import math
import numpy as np
import copy
from concurrent.futures.process import BrokenProcessPool
from pystackreg import StackReg
from Globals import *
from Video import Video
from ProgressBar import *

class Stack:

    SORT_DIFF = 0
    SORT_QUALITY = 1
    SORT_BOTH = 2

    def __init__(self, tmats):
        self.tmats = np.array(tmats, dtype=object)
        self.sortTmats()
        self.stackedImage = None
        self.refBG = None
        self.generateRefBG()
        
    # Checks to see if there will be enough memory to process the image
    def checkMemory(self):
        height, width = g.ui.reference.shape[:2]
        if(not g.ui.checkMemory(w=width*g.drizzleFactor,h=height*g.drizzleFactor)):
            raise MemoryError()
   
    def run(self):
        progress = ProgressBar()
        
        self.stackedImage = None
        i = 0

        # Limit frames
        tmats = list(self.tmats[0:g.limit])
        
        # Sort the frames so that the likelyhood of it being the 'next' frame
        tmats.sort(key=lambda tmat: tmat[0]) 
        
        progress.total = g.limit + 1
        if(g.alignChannels):
            progress.total += 4
        
        progress.setMessage("Stacking Frames", True)
        self.generateRefBG()
            
        gCopy = cloneGlobals()
        futures = []
        try:
            for i in range(0, g.nThreads):
                nFrames = math.ceil(g.limit/g.nThreads)
                frames = tmats[i*nFrames:(i+1)*nFrames]
                if(g.autoCrop):
                    ref = self.refBG
                else:
                    ref = None
                futures.append(g.pool.submit(blendAverage, frames, ref,
                                             g.ui.align.minX, g.ui.align.maxX, g.ui.align.minY, g.ui.align.maxY,
                                             ProgressCounter(progress.counter(i), g.nThreads), gCopy))
            
            for i in range(0, g.nThreads):
                result = futures[i].result()
                if(result is not None):
                    if self.stackedImage is None:
                        self.stackedImage = result
                    else:
                        self.stackedImage += result
        except BrokenProcessPool:
            progress.stop()
            return

        self.stackedImage /= g.limit
        
        # Make sure there wasn't any overshoot from the transformations
        self.stackedImage[self.stackedImage>255] = 255
        self.stackedImage[self.stackedImage<0] = 0
        self.stackedImage = self.stackedImage.astype(np.float32)
        
        if(g.alignChannels):
            progress.setMessage("Aligning RGB", True)
            self.alignChannels(progress)

        progress.stop()
        g.ui.finishedStack()
   
    # Sorts the tmats based on the frameSortMethod
    def sortTmats(self):
        tmats = list(self.tmats)
        if(g.frameSortMethod == Stack.SORT_DIFF):
            tmats.sort(key=lambda tup: tup[2], reverse=True)
        elif(g.frameSortMethod == Stack.SORT_QUALITY):
            tmats.sort(key=lambda tup: tup[3], reverse=True)
        elif(g.frameSortMethod == Stack.SORT_BOTH):
            tmats.sort(key=lambda tup: (tup[2] + tup[3])/2, reverse=True)
        self.tmats = np.array(tmats)
        
    # Creates the background used for transformed images
    def generateRefBG(self):
        (frame, M, diff, sharp) = self.tmats[0]
        ref = g.ui.reference
        self.refBG = transform(ref, None, M,
                               0, 0, 0, 0, g)
        
    # Aligns the RGB channels to help reduce chromatic aberrations
    def alignChannels(self, progress):
        h, w = self.stackedImage.shape[:2]
        gray = cv2.cvtColor(self.stackedImage, cv2.COLOR_BGR2GRAY)
        sr = StackReg(StackReg.TRANSLATION)
        
        for i, C in enumerate(cv2.split(self.stackedImage)):
            M = sr.register(C, gray)
            self.stackedImage[:,:,i] = cv2.warpPerspective(self.stackedImage[:,:,i], M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            progress.setMessage("Aligning RGB", True)
        
# Multiprocess function which sums the given images
def blendAverage(frames, ref, minX, maxX, minY, maxY, progress, gCopy):
    video = Video()
    stackedImage = None
    for c, (frame, M, diff, sharp) in enumerate(frames):
        image = video.getFrame(gCopy.file, frame, gCopy.actualColor())
        image = transform(image, ref, M, 
                          minX, maxX, minY, maxY, gCopy).astype(np.float64)
        if stackedImage is None:
            stackedImage = image
        else:
            stackedImage += image
        progress.count(c, len(frames))
    progress.countExtra()
    return stackedImage
    
# Multiprocess function to transform and save the images to cache
def transform(image, ref, tmat, minX, maxX, minY, maxY, gCopy):
    dst = copy.deepcopy(ref)
    if(ref is not None):
        # Full Frame
        borderMode = cv2.BORDER_TRANSPARENT
    else:
        # Auto Crop
        borderMode = cv2.BORDER_CONSTANT
    i = 0
    I = np.identity(3)
    h, w = image.shape[:2]
    M = tmat.copy()
    
    if(gCopy.drizzleFactor < 1.0):
        # Downscale (need to do it as a separate step since warpPerspective seems to force nearest neighbor when downscaling too much)
        image = cv2.resize(image, (int(w*gCopy.drizzleFactor), int(h*gCopy.drizzleFactor)), interpolation=gCopy.drizzleInterpolation)
        M[0][2] *= gCopy.drizzleFactor # X
        M[1][2] *= gCopy.drizzleFactor # Y
    elif(gCopy.drizzleFactor > 1.0):
        # Upscale
        M[0][2] *= gCopy.drizzleFactor # X
        M[1][2] *= gCopy.drizzleFactor # Y
        T = np.identity(3) # Scale Matrix
        T[0][0] = gCopy.drizzleFactor
        T[1][1] = gCopy.drizzleFactor
        M = M.dot(T) # Apply scale to Transformation

    image = cv2.warpPerspective(image, M, (int(w*gCopy.drizzleFactor), int(h*gCopy.drizzleFactor)), flags=gCopy.drizzleInterpolation, borderMode=borderMode, dst=dst)
    if(ref is None):
        # Auto Crop
        image = image[int(maxY*gCopy.drizzleFactor):int((h+minY)*gCopy.drizzleFactor), 
                      int(maxX*gCopy.drizzleFactor):int((w+minX)*gCopy.drizzleFactor)]
    return image
