import cv2
import numpy as np
from Globals import g

class Stack:

    MEDIAN = "Median"
    AVERAGE = "Average"

    def __init__(self, similarities):
        self.similarities = similarities
        
    def run(self):
        stackedImage = None
        i = 0
        if(g.blendMode == Stack.AVERAGE):
            for frame, diff in self.similarities:
                if(i < g.limit):
                    g.ui.setProgress(i, g.limit-1, "Stacking Frames")
                    image = cv2.imread(frame,1).astype(np.float32) / 255
                    if stackedImage is None:
                        stackedImage = image
                    else:
                        stackedImage += image
                    i += 1
                else:
                    break

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
