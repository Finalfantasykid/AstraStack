import cv2
import numpy as np
import math
import shutil
from pystackreg import StackReg
from Globals import g

class Align:

    def __init__(self, frames):
        self.frames = frames
        self.similarities = []
        self.minX = 0
        self.minY = 0
        self.maxX = 0
        self.maxY = 0
        self.tmats = []
        self.count = 0
        self.total = 0
        self.height, self.width = cv2.imread(self.frames[0]).shape[:2]
        
    # Checks to see if there will be enough memory to process the image
    def checkMemory(self):
        if(not g.ui.checkMemory(w=self.width*g.drizzleFactor,h=self.height*g.drizzleFactor)):
            raise MemoryError()
        
    def run(self):
        def progress(msg):
            self.count += 1
            g.ui.setProgress(self.count, self.total, msg)
        
        threads = []
        g.ui.createListener(progress)
        
        self.total = len(self.frames)*3
        if(g.transformation == -1):
            self.total -= len(self.frames)
        if(g.transformation == -1 and g.drizzleFactor == 1.0):
            self.total -= len(self.frames)
        
        #Drifting
        totalFrames = len(self.frames)
        x1 = g.driftP1[0]
        y1 = g.driftP1[1]
        
        x2 = g.driftP2[0]
        y2 = g.driftP2[1]
        
        dx = x2 - x1
        dy = y2 - y1
        
        fdx, fdy, fdx1, fdy1 = Align.calcDriftDeltas(dx, dy, 1, totalFrames)
        
        # Processing Area
        pa1 = g.processingAreaP1
        pa2 = g.processingAreaP2
        
        if(pa1 != (0,0) and pa2 != (0,0)):
            pa1, pa2 = Align.calcProcessingAreaCoords(pa1, pa2, fdx1, fdy1)
            
        if(abs(pa1[0] - pa2[0]) < 10 or
           abs(pa1[1] - pa2[0]) < 10 or
           pa1[0] > self.width - abs(dx) - 10 or
           pa1[1] > self.height - abs(dy) - 10):
            # If the processing area is too small, just skip it
            pa1 = (0,0)
            pa2 = (0,0)
        
        if((x1 == 0 and y1 == 0) or
           (x2 == 0 and y2 == 0)):
            # Cancel drift if only one point was specified
            dx = 0
            dy = 0
        
        if(dx != 0 and dy != 0):
            self.total += len(self.frames)
        else:
            self.total += 1
            g.ui.childConn.send("Copying Frames")
        futures = []
        for i in range(0, g.nThreads):
            nFrames = math.ceil(len(self.frames)/g.nThreads)
            frames = self.frames[i*nFrames:(i+1)*nFrames]
            futures.append(g.pool.submit(drift, frames, totalFrames, i*nFrames, dx, dy, g.ui.childConn))

        for i in range(0, g.nThreads):
            futures[i].result()
    
        if(g.transformation != -1):
            # Aligning
            futures = []
            for i in range(0, g.nThreads):
                nFrames = math.ceil(len(self.frames)/g.nThreads)
                frames = self.frames[i*nFrames:(i+1)*nFrames]
                futures.append(g.pool.submit(align, frames, g.reference, g.transformation, g.normalize, pa1, pa2, g.ui.childConn))
            
            for i in range(0, g.nThreads):
                self.tmats += futures[i].result()
        else:
            # No Align
            for i, frame in enumerate(self.frames):
                self.tmats.append(np.identity(3))

        # Check how much we need to crop the frames by getting the max and min translations
        for i, frame in enumerate(self.frames):
            M = self.tmats[i]
            
            if(M[0][2] < 0):
                self.minX = min(self.minX, M[0][2])
            else:
                self.maxX = max(self.maxX, M[0][2])
            if(M[1][2] < 0):
                self.minY = min(self.minY, M[1][2])
            else:
                self.maxY = max(self.maxY, M[1][2])
            
            if(pa1 != (0,0) and pa2 != (0,0)):
                # Shift the matrix origin to the processing area, and then shift back
                t1 = np.array([1, 0, -pa1[0], 0, 1, -pa1[1], 0, 0, 1]).reshape((3,3))
                t2 = np.array([1, 0,  pa1[0], 0, 1,  pa1[1], 0, 0, 1]).reshape((3,3))
                self.tmats[i] = t2.dot(M.dot(t1))
        
        self.minX = math.floor(self.minX)
        self.minY = math.floor(self.minY)
        self.maxX = math.ceil(self.maxX)
        self.maxY = math.ceil(self.maxY)
        
        if(g.transformation != -1 or g.drizzleFactor != 1.0):
            # Transforming
            futures = []
            for i in range(0, g.nThreads):
                nFrames = math.ceil(len(self.frames)/g.nThreads)
                frames = self.frames[i*nFrames:(i+1)*nFrames]
                tmats = self.tmats[i*nFrames:(i+1)*nFrames]
                futures.append(g.pool.submit(transform, frames, tmats, 
                                             self.minX, self.maxX, self.minY, self.maxY, 
                                             g.drizzleFactor, g.drizzleInterpolation, 
                                             g.ui.childConn))
            
            for i in range(0, g.nThreads):
                futures[i].result()
            
        if(pa1 != (0,0) and pa2 != (0,0)):
            # Adjust for transformation cropping
            pa1 = (max(0, pa1[0] - self.maxX), max(0, pa1[1] - self.maxY))
            pa2 = (max(0, pa2[0] - self.maxX), max(0, pa2[1] - self.maxY))
            
        # Filtering
        futures = []
        for i in range(0, g.nThreads):
            nFrames = math.ceil(len(self.frames)/g.nThreads)
            frames = self.frames[i*nFrames:(i+1)*nFrames]
            futures.append(g.pool.submit(filter, frames, g.reference, g.normalize, 
                                         pa1, pa2, 
                                         g.drizzleFactor, 
                                         g.ui.childConn))
        
        for i in range(0, g.nThreads):
            self.similarities += futures[i].result()
            
        self.similarities.sort(key=lambda tup: tup[1], reverse=True)
            
        g.ui.setProgress()
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
        
    # returns the processing area coordinates after being drifted
    def calcProcessingAreaCoords(pa1, pa2, fdx1, fdy1):
        pac1 = (int(max(0, pa1[0] - fdx1)), int(max(0, pa1[1] - fdy1)))
        pac2 = (int(max(0, pa2[0] - fdx1)), int(max(0, pa2[1] - fdy1)))
        
        # Account for when points are not drawn in top-left, bottom right
        pa1 = (min(pac1[0], pac2[0]), min(pac1[1], pac2[1]))
        pa2 = (max(pac1[0], pac2[0]), max(pac1[1], pac2[1]))
        return (pa1, pa2)
        
        
# Multiprocess function to drift frames
def drift(frames, totalFrames, startFrame, dx, dy, conn):
    i = startFrame
    for frame in frames:
        if(dx != 0 and dy != 0):
            image = cv2.imread(frame,1)
            
            fdx, fdy, fdx1, fdy1 = Align.calcDriftDeltas(dx, dy, i, totalFrames)
            
            h, w = image.shape[:2]
            image = image[int(fdy1):int(h-fdy), int(fdx1):int(w-fdx)]
        
            cv2.imwrite(frame.replace("frames", "cache"), image)
            i += 1
            conn.send("Drifting Frames")
        else:
            shutil.copyfile(frame, frame.replace("frames", "cache"))           
        
# Multiprocess function to calculation the transform matricies of each image 
def align(frames, reference, transformation, normalize, pa1, pa2, conn):
    ref = cv2.imread(g.tmp + "cache/" + reference + ".png", cv2.IMREAD_GRAYSCALE)
    if(pa1 != (0,0) and pa2 != (0,0)):
        # Processing Area
        ref = ref[pa1[1]:pa2[1],pa1[0]:pa2[0]]
    sr = StackReg(transformation)
    tmats = []
    h, w = ref.shape[:2]
    scaleFactor = min(1.0, (100/h))
    ref = cv2.resize(ref, (int(w*scaleFactor), int(h*scaleFactor)))
    if(normalize):
        ref = cv2.normalize(ref, ref, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    i = 0
    for frame in frames:
        mov = cv2.imread(frame.replace("frames", "cache"), cv2.IMREAD_GRAYSCALE)
        if(pa1 != (0,0) and pa2 != (0,0)):
            # Processing Area
            mov = mov[pa1[1]:pa2[1],pa1[0]:pa2[0]]
        mov = cv2.resize(mov, (int(w*scaleFactor), int(h*scaleFactor)))
        if(normalize):
            mov = cv2.normalize(mov, mov, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        M = sr.register(mov, ref)
        M[0][2] /= scaleFactor # X
        M[1][2] /= scaleFactor # Y
        tmats.append(M)
        conn.send("Aligning Frames")
        i += 1
    return tmats
    
# Multiprocess function to transform and save the images to cache
def transform(frames, tmats, minX, maxX, minY, maxY, drizzleFactor, drizzleInterpolation, conn):
    i = 0
    I = np.identity(3)
    for frame in frames:
        try:
            M = tmats[i]
            image = cv2.imread(frame.replace("frames", "cache"),1)
            h, w = image.shape[:2]
            if(drizzleFactor != 1.0):
                M[0][2] *= drizzleFactor # X
                M[1][2] *= drizzleFactor # Y
                image = cv2.resize(image, (int(w*drizzleFactor), int(h*drizzleFactor)), interpolation=drizzleInterpolation)
            if(not np.array_equal(M, I)):
                image = cv2.warpPerspective(image, M, (int(w*drizzleFactor), int(h*drizzleFactor)), flags=drizzleInterpolation)
                image = image[int(maxY*drizzleFactor):int((h+minY)*drizzleFactor), 
                              int(maxX*drizzleFactor):int((w+minX)*drizzleFactor)]
            cv2.imwrite(frame.replace("frames", "cache"),(image))
        except:
            # Transformation was invalid (resulted infinitely small image)
            pass
        i += 1
        conn.send("Transforming Frames")
    
# Multiprocess function to find the best images (ones closest to the reference frame)
def filter(frames, reference, normalize, pa1, pa2, drizzleFactor, conn):
    similarities = []
    img1 = cv2.imread(g.tmp + "cache/" + reference + ".png", cv2.IMREAD_GRAYSCALE)
    if(pa1 != (0,0) and pa2 != (0,0)):
        # Processing Area
        img1 = img1[int(pa1[1]*drizzleFactor):int(pa2[1]*drizzleFactor),
                    int(pa1[0]*drizzleFactor):int(pa2[0]*drizzleFactor)]
    img1 = cv2.resize(img1, (64,64))
    if(normalize):
        img1 = cv2.normalize(img1, img1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    a, a_norm = calculateNorms(img1)
    for frame in frames:
        img2 = cv2.imread(frame.replace("frames", "cache"), cv2.IMREAD_GRAYSCALE)
        if(pa1 != (0,0) and pa2 != (0,0)):
            # Processing Area
            img2 = img2[int(pa1[1]*drizzleFactor):int(pa2[1]*drizzleFactor),
                        int(pa1[0]*drizzleFactor):int(pa2[0]*drizzleFactor)]
        img2 = cv2.resize(img2, (64,64))
        if(normalize):
            img2 = cv2.normalize(img2, img2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        b, b_norm = calculateNorms(img2)
        diff = np.dot(a / a_norm, b / b_norm)
        similarities.append((frame.replace("frames", "cache"), diff))
        conn.send("Calculating Similarities")
    return similarities

def calculateNorms(img):
    # https://github.com/petermat/image_similarity
    # source: http://www.syntacticbayleaves.com/2008/12/03/determining-image-similarity/
    vector = []
    for pixel_tuple in img.flatten():
        vector.append(np.average(pixel_tuple))
    norm = np.linalg.norm(vector, 2)
    return (vector, norm)
