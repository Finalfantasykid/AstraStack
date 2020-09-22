import cv2
import math
import numpy as np
from pystackreg import StackReg
from Globals import g
from Video import Video

class Stack:

    def __init__(self, tmats):
        self.tmats = tmats
        self.count = 0
        self.total = 0
        self.minX = 0
        self.minY = 0
        self.maxX = 0
        self.maxY = 0
        self.stackedImage = None
        
        # Check how much we need to crop the frames by getting the max and min translations
        for i, tmat in enumerate(self.tmats):
            M = self.tmats[i][1]
            
            if(M[0][2] < 0):
                self.minX = min(self.minX, M[0][2])
            else:
                self.maxX = max(self.maxX, M[0][2])
            if(M[1][2] < 0):
                self.minY = min(self.minY, M[1][2])
            else:
                self.maxY = max(self.maxY, M[1][2])
        
        self.minX = math.floor(self.minX)
        self.minY = math.floor(self.minY)
        self.maxX = math.ceil(self.maxX)
        self.maxY = math.ceil(self.maxY)
        
    # Checks to see if there will be enough memory to process the image
    def checkMemory(self):
        video = Video()
        height, width = video.getFrame(g.file, 0).shape[:2]
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

        # Average Blend Mode
        tmats = self.tmats[0:g.limit]
        self.total = g.limit
        if(g.alignChannels):
            self.total += 4
        futures = []
        for i in range(0, g.nThreads):
            nFrames = math.ceil(g.limit/g.nThreads)
            frames = tmats[i*nFrames:(i+1)*nFrames]
            futures.append(g.pool.submit(blendAverage, frames, g.file,
                                         self.minX, self.maxX, self.minY, self.maxY, 
                                         g.drizzleFactor, g.drizzleInterpolation, g.ui.childConn))
        
        for i in range(0, g.nThreads):
            result = futures[i].result()
            if(result is not None):
                if self.stackedImage is None:
                    self.stackedImage = result
                else:
                    self.stackedImage += result

        self.stackedImage /= g.limit
        
        # Make sure there wasn't any overshoot from the transformations
        self.stackedImage[self.stackedImage>255] = 255
        self.stackedImage[self.stackedImage<0] = 0
        
        if(g.alignChannels):
            g.ui.childConn.send("Aligning RGB")
            self.alignChannels()

        g.ui.finishedStack()
        g.ui.childConn.send("stop")
        
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
def blendAverage(frames, file, minX, maxX, minY, maxY, drizzleFactor, drizzleInterpolation, conn):
    video = Video()
    stackedImage = None
    for frame, M, diff, fd in frames:
        image = video.getFrame(file, frame).astype(np.float32)
        image = transform(image, M, 
                          minX, maxX, minY, maxY, 
                          fd[0], fd[1], fd[2], fd[3], 
                          drizzleFactor, drizzleInterpolation)
        if stackedImage is None:
            stackedImage = image
        else:
            stackedImage += image
        conn.send("Stacking Frames")
    return stackedImage
    
# Multiprocess function to transform and save the images to cache
def transform(image, tmat, minX, maxX, minY, maxY, fdx, fdy, fdx1, fdy1, drizzleFactor, drizzleInterpolation):
    i = 0
    I = np.identity(3)
    h, w = image.shape[:2]
    image = image[int(fdy1):int(h-fdy), int(fdx1):int(w-fdx)]
    h, w = image.shape[:2]
    M = tmat.copy()
    if(drizzleFactor != 1.0):
        M[0][2] *= drizzleFactor # X
        M[1][2] *= drizzleFactor # Y
        T = np.identity(3) # Scale Matrix
        T[0][0] = drizzleFactor
        T[1][1] = drizzleFactor
        M = M.dot(T) # Apply scale to Transformation
    if(not np.array_equal(M, I)):
        image = cv2.warpPerspective(image, M, (int(w*drizzleFactor), int(h*drizzleFactor)), flags=drizzleInterpolation)
        image = image[int(maxY*drizzleFactor):int((h+minY)*drizzleFactor), 
                      int(maxX*drizzleFactor):int((w+minX)*drizzleFactor)]
    return image
