import cv2
import math
import numpy as np
from pystackreg import StackReg
from Globals import g

class Stack:

    def __init__(self, similarities):
        self.similarities = similarities
        self.count = 0
        self.total = 0
        self.stackedImage = None
        
    def run(self):
        def progress(msg):
            self.count += 1
            g.ui.setProgress(self.count, self.total, msg)
        g.ui.createListener(progress)
        
        self.stackedImage = None
        i = 0

        # Average Blend Mode
        similarities = self.similarities[0:g.limit]
        self.total = g.limit
        if(g.alignChannels):
            self.total += 4
        futures = []
        for i in range(0, g.nThreads):
            nFrames = math.ceil(g.limit/g.nThreads)
            frames = similarities[i*nFrames:(i+1)*nFrames]
            futures.append(g.pool.submit(blendAverage, frames, g.ui.childConn))
        
        for i in range(0, g.nThreads):
            result = futures[i].result()
            if(result is not None):
                if self.stackedImage is None:
                    self.stackedImage = result
                else:
                    self.stackedImage += result

        self.stackedImage /= g.limit
        
        if(g.alignChannels):
            g.ui.childConn.send("Aligning RGB")
            self.alignChannels()
        
        cv2.imwrite(g.tmp + "stacked.png",self.stackedImage.astype(np.uint8))
        g.ui.setProgress()
        g.ui.finishedStack()
        g.ui.childConn.send("stop")
        
    # Aligns the RGB channels to help reduce chromatic aberrations
    def alignChannels(self):
        h, w = self.stackedImage.shape[:2]
        gray = cv2.cvtColor(self.stackedImage, cv2.COLOR_BGR2GRAY)
        sr = StackReg(StackReg.TRANSLATION)
        
        minX = 0
        minY = 0
        maxX = 0
        maxY = 0
        for i, C in enumerate(cv2.split(self.stackedImage)):
            M = sr.register(C, gray)
            self.stackedImage[:,:,i] = cv2.warpPerspective(self.stackedImage[:,:,i], M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            g.ui.childConn.send("Aligning RGB")
        
# Multiprocess function which sums the given images
def blendAverage(similarities, conn):
    stackedImage = None
    for frame, diff in similarities:
        image = cv2.imread(frame,1).astype(np.float32)
        if stackedImage is None:
            stackedImage = image
        else:
            stackedImage += image
        conn.send("Stacking Frames")
    return stackedImage
