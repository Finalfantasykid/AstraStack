import cv2
import math
import numpy as np
from Globals import g

class Stack:

    MEDIAN = "Median"
    AVERAGE = "Average"

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
        if(g.blendMode == Stack.AVERAGE):
            # Average Blend Mode
            similarities = self.similarities[0:g.limit]
            self.total = g.limit
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
        elif(g.blendMode == Stack.MEDIAN):
            # Median Blend Mode
            stacked = []
            for frame, diff in self.similarities:
                if(i < g.limit):
                    g.ui.setProgress(i, g.limit-1, "Stacking Frames")
                    image = cv2.imread(frame,1)
                    stacked.append(image)
                    i += 1
                else:
                    break
                    
            stacked = np.array(stacked)
            self.stackedImage = np.median(stacked, axis=0)
        
        cv2.imwrite(g.tmp + "stacked.png",self.stackedImage.astype(np.uint8))
        g.ui.setProgress()
        g.ui.finishedStack()
        g.ui.childConn.send("stop")
        
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
