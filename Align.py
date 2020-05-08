import cv2
import numpy as np
import math
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
        
    def run(self):
        def progress(msg):
            self.count += 1
            g.ui.setProgress(self.count, self.total, msg)
        
        threads = []
        g.ui.createListener(progress)
        self.total = len(self.frames)*4
        
        #Drifting
        dx = g.driftP2[0] - g.driftP1[0]
        dy = g.driftP2[1] - g.driftP1[1]
        
        futures = []
        for i in range(0, g.nThreads):
            nFrames = math.ceil(len(self.frames)/g.nThreads)
            frames = self.frames[i*nFrames:(i+1)*nFrames]
            futures.append(g.pool.submit(drift, frames, len(self.frames), i*nFrames, dx, dy, g.ui.childConn))

        for i in range(0, g.nThreads):
            futures[i].result()
    
        # Aligning
        futures = []
        for i in range(0, g.nThreads):
            nFrames = math.ceil(len(self.frames)/g.nThreads)
            frames = self.frames[i*nFrames:(i+1)*nFrames]
            futures.append(g.pool.submit(align, frames, g.reference, g.transformation, g.normalize, g.ui.childConn))
        
        for i in range(0, g.nThreads):
            self.tmats += futures[i].result()
        
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
            
        self.minX = math.floor(self.minX)
        self.minY = math.floor(self.minY)
        self.maxX = math.ceil(self.maxX)
        self.maxY = math.ceil(self.maxY)
        
        # Transforming
        futures = []
        for i in range(0, g.nThreads):
            nFrames = math.ceil(len(self.frames)/g.nThreads)
            frames = self.frames[i*nFrames:(i+1)*nFrames]
            tmats = self.tmats[i*nFrames:(i+1)*nFrames]
            futures.append(g.pool.submit(transform, frames, tmats, self.minX, self.maxX, self.minY, self.maxY, g.ui.childConn))
        
        for i in range(0, g.nThreads):
            futures[i].result()
            
        # Filtering
        futures = []
        for i in range(0, g.nThreads):
            nFrames = math.ceil(len(self.frames)/g.nThreads)
            frames = self.frames[i*nFrames:(i+1)*nFrames]
            futures.append(g.pool.submit(filter, frames, g.reference, g.normalize, g.ui.childConn))
        
        for i in range(0, g.nThreads):
            self.similarities += futures[i].result()
            
        self.similarities.sort(key=lambda tup: tup[1], reverse=True)
            
        g.ui.setProgress()
        g.ui.finishedAlign()
        g.ui.childConn.send("stop")
        
# Multiprocess function to drift frames
def drift(frames, totalFrames, startFrame, dx, dy, conn):
    i = startFrame
    for frame in frames:
        image = cv2.imread(frame,1)
        if(dx != 0 and dy != 0):
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
            
            h, w = image.shape[:2]
            image = image[int(fdy1):int(h-fdy), int(fdx1):int(w-fdx)]
        
        cv2.imwrite(frame.replace("frames", "cache"), image)
        i += 1
        if(dx != 0 and dy != 0):
            conn.send("Drifting Frames")
        else:
            conn.send("Copying Frames")
        
# Multiprocess function to calculation the transform matricies of each image 
def align(frames, reference, transformation, normalize, conn):
    ref = cv2.imread(g.tmp + "cache/" + reference + ".png", cv2.IMREAD_GRAYSCALE)
    sr = StackReg(transformation)
    tmats = []
    h, w = ref.shape[:2]
    scaleFactor = min(1.0, (100/h))
    ref = cv2.resize(ref, (int(w*scaleFactor), int(h*scaleFactor)), interpolation=cv2.INTER_LINEAR)
    if(normalize):
        ref = cv2.normalize(ref, ref, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    for frame in frames:
        mov = cv2.imread(frame.replace("frames", "cache"), cv2.IMREAD_GRAYSCALE)
        mov = cv2.resize(mov, (int(w*scaleFactor), int(h*scaleFactor)), interpolation=cv2.INTER_LINEAR)
        if(normalize):
            mov = cv2.normalize(mov, mov, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        M = sr.register(mov, ref)
        M[0][2] /= scaleFactor # X
        M[1][2] /= scaleFactor # Y
        tmats.append(M)
        conn.send("Aligning Frames")
    return tmats
    
# Multiprocess function to transform and save the images to cache
def transform(frames, tmats, minX, maxX, minY, maxY, conn):
    i = 0
    for frame in frames:
        M = tmats[i]
        image = cv2.imread(frame.replace("frames", "cache"),1).astype(np.float32) / 255
        w, h, _ = image.shape
        image = cv2.warpPerspective(image, M, (h, w))
        h, w = image.shape[:2]
        image = image[maxY:h+minY, maxX:w+minX]
        cv2.imwrite(frame.replace("frames", "cache"),(image*255).astype(np.uint8))
        i += 1
        conn.send("Transforming Frames")
    
# Multiprocess function to find the best images (ones closest to the reference frame)
def filter(frames, reference, normalize, conn):
    similarities = []
    img1 = cv2.resize(cv2.imread(g.tmp + "cache/" + reference + ".png", cv2.IMREAD_GRAYSCALE), (64,64))
    if(normalize):
        img1 = cv2.normalize(img1, img1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    a, a_norm = calculateNorms(img1)
    for frame in frames:
        img2 = cv2.resize(cv2.imread(frame.replace("frames", "cache"), cv2.IMREAD_GRAYSCALE), (64,64))
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
