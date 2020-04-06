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
        
    def run(self):
        def progress(msg):
            self.count += 1
            g.ui.setProgress(self.count, self.total, msg)
        g.ui.createListener(progress)
        
        stackedImage = None
        i = 0
        if(g.blendMode == Stack.AVERAGE):
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
                    if stackedImage is None:
                        stackedImage = result
                    else:
                        stackedImage += result

            stackedImage /= g.limit
            stackedImage = (stackedImage*255).astype(np.uint8)
        elif(g.blendMode == Stack.MEDIAN):
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
            stackedImage = np.median(stacked, axis=0)
        
        cv2.imwrite(g.tmp + "stacked.png",stackedImage)
        g.ui.setProgress()
        g.ui.finishedStack()
        g.ui.childConn.send("stop")
        
def blendAverage(similarities, conn):
    stackedImage = None
    for frame, diff in similarities:
        image = cv2.imread(frame,1).astype(np.float32) / 255
        if stackedImage is None:
            stackedImage = image
        else:
            stackedImage += image
        conn.send("Stacking Frames")
    return stackedImage
