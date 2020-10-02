import cv2
import math
import numpy as np
import copy
from pystackreg import StackReg
from Globals import g
from Video import Video

class Align:

    def __init__(self, frames):
        self.frames = frames
        self.tmats = [] # (frame, M, diff, (fdx, fdy, fdx1, fdy1))
        self.count = 0
        self.total = 0
        self.minX = 0
        self.minY = 0
        self.maxX = 0
        self.maxY = 0
        video = Video()
        self.height, self.width = video.getFrame(g.file, 0).shape[:2]
        
    def run(self):
        def progress(msg):
            self.count += 1
            g.ui.setProgress(self.count, self.total, msg)
        g.ui.createListener(progress)
        self.count = 0
        self.total = len(self.frames)
        
        #Drifting
        totalFrames = len(self.frames)
        
        x1 = g.driftP1[0]
        y1 = g.driftP1[1]
        
        x2 = g.driftP2[0]
        y2 = g.driftP2[1]
        
        dx = x2 - x1
        dy = y2 - y1
        
        fdx, fdy, fdx1, fdy1 = Align.calcDriftDeltas(dx, dy, 1, totalFrames)
        
        # Area of Interest
        aoi1 = g.areaOfInterestP1
        aoi2 = g.areaOfInterestP2
        
        if(aoi1 != (0,0) and aoi2 != (0,0)):
            aoi1, aoi2 = Align.calcAreaOfInterestCoords(aoi1, aoi2, fdx1, fdy1)
            
        if(abs(aoi1[0] - aoi2[0]) < 10 or
           abs(aoi1[1] - aoi2[0]) < 10 or
           aoi1[0] > self.width - abs(dx) - 10 or
           aoi1[1] > self.height - abs(dy) - 10):
            # If the Area of Interest is too small, just skip it
            aoi1 = (0,0)
            aoi2 = (0,0)
        
        if((x1 == 0 and y1 == 0) or
           (x2 == 0 and y2 == 0)):
            # Cancel drift if only one point was specified
            dx = 0
            dy = 0

        # Aligning
        futures = []
        if(int(g.reference) in self.frames):
            referenceIndex = self.frames.index(int(g.reference))
        else:
            referenceIndex = int(g.reference) - g.startFrame
        
        for i in range(0, g.nThreads):
            nFrames = math.ceil(len(self.frames)/g.nThreads)
            frames = self.frames[i*nFrames:(i+1)*nFrames]
            futures.append(g.pool.submit(align, frames, g.file, self.frames[referenceIndex], referenceIndex, 
                                         g.transformation, g.normalize, totalFrames, i*nFrames, dx, dy, aoi1, aoi2, g.ui.childConn))
        
        for i in range(0, g.nThreads):
            tmats, minX, minY, maxX, maxY = futures[i].result()
            self.tmats += tmats
            self.minX = math.floor(min(self.minX, minX))
            self.maxX = math.ceil(max(self.maxX, maxX))
            self.minY = math.floor(min(self.minY, minY))
            self.maxY = math.ceil(max(self.maxY, maxY))
            
        self.tmats.sort(key=lambda tup: tup[2], reverse=True)

        g.ui.finishedAlign()
        g.ui.childConn.send("stop")
        
    # returns a list of the delta values for the drift points
    def calcDriftDeltas(dx, dy, i, totalFrames):
        fdx = dx*i/totalFrames
        fdy = dy*i/totalFrames
        
        if(dx > 0):
            fdx = dx - fdx
        if(dy > 0):
            fdy = dy - fdy
        
        fdx1 = abs(dx - fdx)
        fdy1 = abs(dy - fdy)
            
        fdx = abs(fdx)
        fdy = abs(fdy)
        
        return (fdx, fdy, fdx1, fdy1)
        
    # returns the Area of Interest coordinates after being drifted
    def calcAreaOfInterestCoords(aoi1, aoi2, fdx1, fdy1):
        aoic1 = (int(max(0, aoi1[0] - fdx1)), int(max(0, aoi1[1] - fdy1)))
        aoic2 = (int(max(0, aoi2[0] - fdx1)), int(max(0, aoi2[1] - fdy1)))
        
        # Account for when points are not drawn in top-left, bottom right
        aoi1 = (min(aoic1[0], aoic2[0]), min(aoic1[1], aoic2[1]))
        aoi2 = (max(aoic1[0], aoic2[0]), max(aoic1[1], aoic2[1]))
        return (aoi1, aoi2)     
        
# Multiprocess function to calculation the transform matricies of each image 
def align(frames, file, reference, referenceIndex, transformation, normalize, totalFrames, startFrame, dx, dy, aoi1, aoi2, conn):
    i = startFrame
    tmats = []
    minX = 0
    minY = 0
    maxX = 0
    maxY = 0
    video = Video()
    ref = cv2.cvtColor(video.getFrame(file, reference), cv2.COLOR_BGR2GRAY)
    h1, w1 = ref.shape[:2]
    if(dx != 0 and dy != 0):
        # Drift
        fdx, fdy, fdx1, fdy1 = Align.calcDriftDeltas(dx, dy, referenceIndex, totalFrames)   
        ref = ref[int(fdy1):int(h1-fdy), int(fdx1):int(w1-fdx)]
    
    if(aoi1 != (0,0) and aoi2 != (0,0)):
        # Area of Interest
        ref = ref[aoi1[1]:aoi2[1],aoi1[0]:aoi2[0]]
    if(transformation != -1):
        sr = StackReg(transformation)
    else:
        sr = None
    h, w = ref.shape[:2]
    scaleFactor = min(1.0, (100/h))
    ref = cv2.resize(ref, (int(w*scaleFactor), int(h*scaleFactor)))
    if(normalize):
        ref = cv2.normalize(ref, ref, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    a, a_norm = calculateNorms(cv2.resize(ref, (64,64)))
    for frame in frames:
        try:
            fdx  = 0
            fdy  = 0
            fdx1 = 0
            fdy1 = 0
            mov = cv2.cvtColor(video.getFrame(file, frame), cv2.COLOR_BGR2GRAY)
            if(dx != 0 and dy != 0):
                # Drift
                fdx, fdy, fdx1, fdy1 = Align.calcDriftDeltas(dx, dy, i, totalFrames)   
                mov = mov[int(fdy1):int(h1-fdy), int(fdx1):int(w1-fdx)]
            
            if(aoi1 != (0,0) and aoi2 != (0,0)):
                # Area of Interest
                mov = mov[aoi1[1]:aoi2[1],aoi1[0]:aoi2[0]]
            mov = cv2.resize(mov, (int(w*scaleFactor), int(h*scaleFactor)))
            if(normalize):
                mov = cv2.normalize(mov, mov, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            if(sr is not None):
                M = sr.register(mov, ref)
            else:
                M = np.identity(3)
            
            # Apply transformation to small version to check similarity to reference
            dst = copy.deepcopy(ref)
            mov = cv2.warpPerspective(mov, M, (int(w*scaleFactor), int(h*scaleFactor)), borderMode=cv2.BORDER_TRANSPARENT, dst=dst)
            b, b_norm = calculateNorms(cv2.resize(mov, (64,64)))
            diff = np.dot(a / a_norm, b / b_norm)
            
            M[0][2] /= scaleFactor # X
            M[1][2] /= scaleFactor # Y
            
            # Used for auto-crop
            if(M[0][2] < 0):
                minX = min(minX, M[0][2])
            else:
                maxX = max(maxX, M[0][2])
            if(M[1][2] < 0):
                minY = min(minY, M[1][2])
            else:
                maxY = max(maxY, M[1][2])
            
            if(aoi1 != (0,0) and aoi2 != (0,0)):
                # Shift the matrix origin to the Area of Interest, and then shift back
                t1 = np.array([1, 0, -aoi1[0], 0, 1, -aoi1[1], 0, 0, 1]).reshape((3,3))
                t2 = np.array([1, 0,  aoi1[0], 0, 1,  aoi1[1], 0, 0, 1]).reshape((3,3))
                M = t2.dot(M.dot(t1))
            
            tmats.append((frame, M, diff, (fdx, fdy, fdx1, fdy1)))
        except Exception as e:
            print(e)
        conn.send("Aligning Frames")
        i += 1
    return (tmats, minX, minY, maxX, maxY)

def calculateNorms(img):
    # https://github.com/petermat/image_similarity
    # source: http://www.syntacticbayleaves.com/2008/12/03/determining-image-similarity/
    vector = []
    for pixel_tuple in img.flatten():
        vector.append(np.average(pixel_tuple))
    norm = np.linalg.norm(vector, 2)
    return (vector, norm)
