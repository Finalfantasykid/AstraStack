import cv2
import math
import numpy as np
import copy
from concurrent.futures.process import BrokenProcessPool
from pystackreg import StackReg
from Globals import g
from Video import Video

class Stack:

    def __init__(self, tmats):
        self.tmats = tmats
        self.count = 0
        self.total = 0
        self.stackedImage = None
        self.refBG = None
        self.generateRefBG()
        
    # Checks to see if there will be enough memory to process the image
    def checkMemory(self):
        video = Video()
        height, width = video.getFrame(g.file, 0, (g.colorMode or g.guessedColorMode), fast=True).shape[:2]
        if(not g.ui.checkMemory(w=width*g.drizzleFactor,h=height*g.drizzleFactor)):
            raise MemoryError()
   
    def run(self):
        def progress(msg):
            self.count += 1
            g.ui.setProgress(self.count, self.total, msg)
            
        g.ui.createListener(progress)
        self.count = 0
        
        self.stackedImage = None
        i = 0

        # Limit frames
        tmats = self.tmats[0:g.limit]
        
        # Sort the frames so that the likelyhood of it being the 'next' frame
        tmats.sort(key=lambda tmat: tmat[0]) 
        
        self.total = g.limit
        if(g.alignChannels):
            self.total += 4
            
        self.generateRefBG()
            
        futures = []
        try:
            for i in range(0, g.nThreads):
                nFrames = math.ceil(g.limit/g.nThreads)
                frames = tmats[i*nFrames:(i+1)*nFrames]
                if(g.autoCrop):
                    ref = self.refBG
                else:
                    ref = None
                futures.append(g.pool.submit(blendAverage, frames, g.file, ref,
                                             g.ui.align.minX, g.ui.align.maxX, g.ui.align.minY, g.ui.align.maxY, 
                                             g.drizzleFactor, g.drizzleInterpolation,
                                             (g.colorMode or g.guessedColorMode), g.ui.childConn))
            
            for i in range(0, g.nThreads):
                result = futures[i].result()
                if(result is not None):
                    if self.stackedImage is None:
                        self.stackedImage = result
                    else:
                        self.stackedImage += result
        except BrokenProcessPool:
            g.ui.childConn.send("stop")
            return

        self.stackedImage /= g.limit
        
        # Make sure there wasn't any overshoot from the transformations
        self.stackedImage[self.stackedImage>255] = 255
        self.stackedImage[self.stackedImage<0] = 0
        self.stackedImage = self.stackedImage.astype(np.float32)
        
        if(g.alignChannels):
            g.ui.childConn.send("Aligning RGB")
            self.alignChannels()

        g.ui.finishedStack()
        g.ui.childConn.send("stop")
        
    # Creates the background used for transformed images
    def generateRefBG(self):
        video = Video()
        (frame, M, diff) = self.tmats[0]
        ref = video.getFrame(g.file, frame, (g.colorMode or g.guessedColorMode)).astype(np.float32)
        self.refBG = transform(ref, None, np.identity(3),
                               0, 0, 0, 0,
                               g.drizzleFactor, g.drizzleInterpolation)
        
    # Aligns the RGB channels to help reduce chromatic aberrations
    def alignChannels(self):
        h, w = self.stackedImage.shape[:2]
        gray = cv2.cvtColor(self.stackedImage, cv2.COLOR_BGR2GRAY)
        sr = StackReg(StackReg.TRANSLATION)
        
        for i, C in enumerate(cv2.split(self.stackedImage)):
            M = sr.register(C, gray)
            self.stackedImage[:,:,i] = cv2.warpPerspective(self.stackedImage[:,:,i], M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            g.ui.childConn.send("Aligning RGB")
        
# Multiprocess function which sums the given images
def blendAverage(frames, file, ref, minX, maxX, minY, maxY, drizzleFactor, drizzleInterpolation, colorMode, conn):
    video = Video()
    stackedImage = None
    for frame, M, diff in frames:
        image = video.getFrame(file, frame, colorMode)
        image = transform(image, ref, M, 
                          minX, maxX, minY, maxY,
                          drizzleFactor, drizzleInterpolation).astype(np.float64)
        if stackedImage is None:
            stackedImage = image
        else:
            stackedImage += image
        conn.send("Stacking Frames")
    return stackedImage
    
# Multiprocess function to transform and save the images to cache
def transform(image, ref, tmat, minX, maxX, minY, maxY, drizzleFactor, drizzleInterpolation):
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
    
    if(drizzleFactor < 1.0):
        # Downscale (need to do it as a separate step since warpPerspective seems to force nearest neighbor when downscaling too much)
        image = cv2.resize(image, (int(w*drizzleFactor), int(h*drizzleFactor)), interpolation=drizzleInterpolation)
        M[0][2] *= drizzleFactor # X
        M[1][2] *= drizzleFactor # Y
    elif(drizzleFactor > 1.0):
        # Upscale
        M[0][2] *= drizzleFactor # X
        M[1][2] *= drizzleFactor # Y
        T = np.identity(3) # Scale Matrix
        T[0][0] = drizzleFactor
        T[1][1] = drizzleFactor
        M = M.dot(T) # Apply scale to Transformation

    image = cv2.warpPerspective(image, M, (int(w*drizzleFactor), int(h*drizzleFactor)), flags=drizzleInterpolation, borderMode=borderMode, dst=dst)
    if(ref is None):
        # Auto Crop
        image = image[int(maxY*drizzleFactor):int((h+minY)*drizzleFactor), 
                      int(maxX*drizzleFactor):int((w+minX)*drizzleFactor)]
    return image
