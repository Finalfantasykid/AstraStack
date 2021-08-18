import cv2
import math
import numpy as np
from concurrent.futures.process import BrokenProcessPool
from pystackreg import StackReg
from Globals import *
from Video import Video
from ProgressBar import *

class Align:

    DRIFT_NONE = 0
    DRIFT_GRAVITY = 1
    DRIFT_MANUAL = 2

    def __init__(self, frames):
        self.frames = frames
        self.tmats = [] # (frame, M, diff, sharp)
        self.minX = 0
        self.minY = 0
        self.maxX = 0
        self.maxY = 0
        self.height, self.width = g.ui.video.getFrame(g.file, 0, g.actualColor()).shape[:2]
        
    def run(self):
        progress = ProgressBar()
        progress.total = len(self.frames)
        
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
            
        try:
            progress.setMessage("Aligning Frames")
            ref = cv2.cvtColor(g.ui.video.getFrame(g.file, self.frames[referenceIndex], g.actualColor()), cv2.COLOR_BGR2GRAY)
            gCopy = cloneGlobals()
            for i in range(0, g.nThreads):
                nFrames = math.ceil(len(self.frames)/g.nThreads)
                frames = self.frames[i*nFrames:(i+1)*nFrames]
                futures.append(g.pool.submit(align, frames, ref, referenceIndex, 
                                             totalFrames, i*nFrames, dx, dy, aoi1, aoi2, 
                                             ProgressCounter(progress.counter(i), g.nThreads), gCopy))
            
            for i in range(0, g.nThreads):
                tmats, minX, minY, maxX, maxY = futures[i].result()
                self.tmats += tmats
                self.minX = math.floor(min(self.minX, minX))
                self.maxX = math.ceil(max(self.maxX, maxX))
                self.minY = math.floor(min(self.minY, minY))
                self.maxY = math.ceil(max(self.maxY, maxY))
        except BrokenProcessPool:
            progress.stop()
            return
        
        self.tmats = np.array(self.tmats, dtype=object)
        self.tmats[:,3] = g.ui.video.sharps[g.startFrame:g.endFrame+1]
        self.tmats = self.tmats[self.tmats[:,2]!=False]
        
        diffs = np.float64(self.tmats[:,2])
        sharps = np.float64(self.tmats[:,3])
        
        if(len(self.tmats) > 1):
            self.tmats[:,2] = cv2.normalize(diffs, diffs, 0, 1, cv2.NORM_MINMAX)
            self.tmats[:,3] = cv2.normalize(sharps, sharps, 0, 1, cv2.NORM_MINMAX)

        progress.stop()
        g.ui.finishedAlign()
        
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
        
    # Returns the Area of Interest coordinates after being drifted
    def calcAreaOfInterestCoords(aoi1, aoi2, fdx1, fdy1):
        aoic1 = (int(max(0, aoi1[0] - fdx1)), int(max(0, aoi1[1] - fdy1)))
        aoic2 = (int(max(0, aoi2[0] - fdx1)), int(max(0, aoi2[1] - fdy1)))
        
        # Account for when points are not drawn in top-left, bottom right
        aoi1 = (min(aoic1[0], aoic2[0]), min(aoic1[1], aoic2[1]))
        aoi2 = (max(aoic1[0], aoic2[0]), max(aoic1[1], aoic2[1]))
        return (aoi1, aoi2)
    
    # Returns new min/max X/Y shifts
    def calcMinMax(M, minX, maxX, minY, maxY):
        if(M[0][2] < 0):
            minX = min(minX, M[0][2])
        else:
            maxX = max(maxX, M[0][2])
        if(M[1][2] < 0):
            minY = min(minY, M[1][2])
        else:
            maxY = max(maxY, M[1][2])
        return (minX, maxX, minY, maxY)
        
# Multiprocess function to calculation the transform matricies of each image 
def align(frames, ref, referenceIndex, totalFrames, startFrame, dx, dy, aoi1, aoi2, progress, gCopy):
    i = startFrame
    tmats = []
    minX = minY = maxX = maxY = 0
    video = Video()
    
    # Load Reference
    refOrig = ref
    h1, w1 = ref.shape[:2]
    
    # Drift
    rfdx, rfdy, rfdx1, rfdy1 = Align.calcDriftDeltas(dx, dy, referenceIndex, totalFrames)   
    ref = ref[int(rfdy1):int(h1-rfdy), int(rfdx1):int(w1-rfdx)]
    
    # Center of Mass
    if(gCopy.driftType == Align.DRIFT_GRAVITY):
        (cx, cy) = centerOfMass(ref)
        C = np.float32([[1, 0, int(w1/2) - int(cx)], 
                        [0, 1, int(h1/2) - int(cy)]])

        ref = cv2.warpAffine(ref, C, (w1, h1), flags=cv2.INTER_NEAREST)
        refOrig = cv2.warpAffine(refOrig, C, (w1, h1), flags=cv2.INTER_NEAREST)
        
    # Area of Interest
    ref = cropAreaOfInterest(ref, aoi1, aoi2)
    refOrig = cropAreaOfInterest(refOrig, aoi1, aoi2, rfdx1, rfdy1)
    
    if(gCopy.transformation != -1):
        sr = StackReg(gCopy.transformation)
    else:
        sr = None
        
    h, w = ref.shape[:2]
    scaleFactor = min(1.0, (100/h))
    ref = cv2.resize(ref, (int(w*scaleFactor), int(h*scaleFactor)))
    refOrig = cv2.resize(refOrig, (64, 64))
    
    if(gCopy.normalize):
        ref = normalizeImg(ref)
        refOrig = normalizeImg(refOrig)
    
    for c, frame in enumerate(frames):
        try:
            # Load Frame
            movOrig = mov = cv2.cvtColor(video.getFrame(gCopy.file, frame, gCopy.actualColor()), cv2.COLOR_BGR2GRAY)

            # Drift
            fdx, fdy, fdx1, fdy1 = Align.calcDriftDeltas(dx, dy, i, totalFrames)   
            mov = mov[int(fdy1):int(h1-fdy), int(fdx1):int(w1-fdx)]
            
            # Center of Mass
            if(gCopy.driftType == Align.DRIFT_GRAVITY):
                (cx, cy) = centerOfMass(mov)
                C = np.float32([[1, 0, int(w1/2) - int(cx)], 
                                [0, 1, int(h1/2) - int(cy)]])
                mov = cv2.warpAffine(mov, C, (w1, h1), flags=cv2.INTER_NEAREST)
            
            # Area of Interest
            mov = cropAreaOfInterest(mov, aoi1, aoi2)
            
            # Resize
            mov = cv2.resize(mov, (int(w*scaleFactor), int(h*scaleFactor)))
            
            if(gCopy.normalize):
                mov = normalizeImg(mov)

            # Stack Registration
            if(sr is not None):
                M = sr.register(mov, ref)
            else:
                M = np.identity(3)
            
            # Scale back up
            M[0][2] /= scaleFactor # X
            M[1][2] /= scaleFactor # Y

            # Only add if the stack registration actually converged
            if(abs(M[0][2]) >= w or abs(M[1][2]) >= h):
                tmats.append([frame, M, False, 0])
                progress.count(c, len(frames))
                continue

            # Shift the matrix origin to the Area of Interest, and then shift back
            M = shiftOrigin(M, aoi1[0], aoi1[1])
            
            # Add center of mass transform
            if(gCopy.driftType == Align.DRIFT_GRAVITY):
                M[0][2] += int(w1/2) - int(cx)
                M[1][2] += int(h1/2) - int(cy)
                M = shiftOrigin(M, -(int(w1/2) - int(cx)), -(int(h1/2) - int(cy)))
                
            # Add drift transform
            M[0][2] -= int(fdx1) - int(rfdx1)
            M[1][2] -= int(fdy1) - int(rfdy1)
            M = shiftOrigin(M, int(fdx1), int(fdy1))
            
            # Apply transformation to small version to check similarity to reference
            movOrig = cv2.warpPerspective(movOrig, M, (w1, h1), borderMode=cv2.BORDER_REPLICATE)

            if(aoi1 != (0,0) and aoi2 != (0,0)):
                # Area of Interest
                movOrig = cropAreaOfInterest(movOrig, aoi1, aoi2, rfdx1, rfdy1)
                xFactor = None
                yFactor = None
            else:
                xFactor = 64/movOrig.shape[1]
                yFactor = 64/movOrig.shape[0]
            movOrig = cv2.resize(movOrig, (64, 64))
            
            if(gCopy.normalize): 
                movOrig = normalizeImg(movOrig)
                
            # Similarity
            diff = calculateDiff(refOrig, movOrig, xFactor, yFactor, M, i)

            # Make sure it isn't going to autocrop to point where the image is too small
            if((aoi1 == (0,0) and aoi2 == (0,0) and abs(M[0][2]) < w1/3 and abs(M[1][2]) < h1/3) or 
               (aoi1 != (0,0) and aoi2 != (0,0) and abs(M[0][2]) < min(w1/3, w1/2 - w/2) and abs(M[1][2]) < min(h1/3, h1/2 - h/2))):
                # Used for auto-crop
                minX, maxX, minY, maxY = Align.calcMinMax(M, minX, maxX, minY, maxY)
            
            tmats.append([frame, M, diff, 0])    
        except Exception as e:
            print(e)
        progress.count(c, len(frames))
        i += 1
    progress.countExtra()
    return (tmats, minX, minY, maxX, maxY)

# Calculates the center of mass (coordinates) of the image
def centerOfMass(img):
    y = range(0, img.shape[0])
    x = range(0, img.shape[1])

    (X,Y) = np.meshgrid(x,y)

    img = img.astype(np.float32)
    img -= 15
    img[img<0] = 0
    
    img = img**2

    imgSum = max(1,img.sum())

    x_coord = (X*img).sum() / imgSum
    y_coord = (Y*img).sum() / imgSum
    
    return (x_coord, y_coord)

# Crop image for Area of Interest
def cropAreaOfInterest(img, aoi1, aoi2, fdx=0, fdy=0):
    if(aoi1 != (0,0) and aoi2 != (0,0)):
        img = img[int(aoi1[1]+fdy):int(aoi2[1]+fdy),
                  int(aoi1[0]+fdx):int(aoi2[0]+fdx)]
    return img

# Shifts the position of the origin for rotations etc, and then back to where it was initially
def shiftOrigin(M, x, y):
    if(x != 0 or y != 0):
        t1 = np.array([1, 0, -x, 0, 1, -y, 0, 0, 1]).reshape((3,3))
        t2 = np.array([1, 0,  x, 0, 1,  y, 0, 0, 1]).reshape((3,3))
        M = t2.dot(M.dot(t1))
    return M

# Nomalizes pixel values between 0 and 255
def normalizeImg(img):
    return cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Returns the similarity between the two images
def calculateDiff(ref, mov, xFactor, yFactor, M, i):
    # Simulate smaller auto-crop
    if(xFactor != None and yFactor != None):
        minX, maxX, minY, maxY = Align.calcMinMax(M, 0, 0, 0, 0)
        minX *= xFactor
        maxX *= xFactor
        minY *= yFactor
        maxY *= yFactor
        h, w = ref.shape[:2]

        ref = ref[int(maxY):int((h+minY)), int(maxX):int((w+minX))]
        mov = mov[int(maxY):int((h+minY)), int(maxX):int((w+minX))]
    
    h, w = ref.shape[:2]

    # Modified Means Squared Difference
    diff = 1 - np.sum((cv2.absdiff(mov.astype(np.float32)/255, ref.astype(np.float32)/255)) ** 2)/(h*w)
    return diff
