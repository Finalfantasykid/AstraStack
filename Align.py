import cv2
import numpy as np
import math
from PIL import Image
from multiprocessing import Manager, Process, Value, Lock
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
        self.count = Value('i', 0)
        self.lock = Lock()
        self.total = 0
        
    def run(self):
        def progress(msg):
            g.ui.setProgress(self.count.value, self.total, msg)
            
        threads = []
        g.ui.createListener(progress)
        self.total = len(self.frames)*3
        
        #Drifting
        if(g.driftP1 != (0, 0) or g.driftP1 != (0, 0)):
            self.total += len(self.frames)
            dx = g.driftP2[0] - g.driftP1[0]
            dy = g.driftP2[1] - g.driftP1[1]
            
            i = 0
            for frame in self.frames:
                image = cv2.imread(frame,1)
                fdx = dx*i/len(self.frames)
                fdy = dy*i/len(self.frames)
                
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
                
                cv2.imwrite(frame, image)
                i += 1
                self.count.value += 1
                g.ui.childConn.send("Drifting Frames")
    
        # Aligning
        manager = Manager()
        rets = manager.dict()
        self.tmats = []
        for i in range(0, g.nThreads):
            nFrames = math.ceil(len(self.frames)/g.nThreads)
            frames = self.frames[i*nFrames:(i+1)*nFrames]
            thread = Process(target=self.align, args=(i, rets, frames, g.ui.childConn, ))
            threads.append(thread)
        
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        for i in range(0, g.nThreads):
            self.tmats += rets[i]

        M = np.identity(3);
        for frame in self.frames:
            M = np.copy(self.tmats[i])
            
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
        threads = []
        for i in range(0, g.nThreads):
            nFrames = math.ceil(len(self.frames)/g.nThreads)
            frames = self.frames[i*nFrames:(i+1)*nFrames]
            tmats = self.tmats[i*nFrames:(i+1)*nFrames]
            thread = Process(target=self.transform, args=(frames, tmats, g.ui.childConn, ))
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
            
        # Filtering
        threads = []
        rets = manager.dict()
        for i in range(0, g.nThreads):
            nFrames = math.ceil(len(self.frames)/g.nThreads)
            frames = self.frames[i*nFrames:(i+1)*nFrames]
            thread = Process(target=self.filter, args=(i, rets, frames, g.ui.childConn))
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        for i in range(0, g.nThreads):
            self.similarities += rets[i]
            
        self.similarities.sort(key=lambda tup: tup[1], reverse=True)
            
        g.ui.setProgress()
        g.ui.finishedAlign()
        g.ui.childConn.send("stop")

    # Multiprocess function to calculation the transform matricies of each image 
    def align(self, processId, rets, frames, conn):
        ref = cv2.imread("frames/" + g.reference + ".png", cv2.IMREAD_GRAYSCALE)
        sr = StackReg(g.transformation)
        tmats = []
        h, w = ref.shape[:2]
        scaleFactor = 0.5
        ref = cv2.resize(ref, (int(w*scaleFactor), int(h*scaleFactor)), interpolation=cv2.INTER_NEAREST)
        for frame in frames:
            mov = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
            mov = cv2.resize(mov, (int(w*scaleFactor), int(h*scaleFactor)), interpolation=cv2.INTER_NEAREST)
            
            M = sr.register(mov, ref)
            M[0][2] /= scaleFactor # X
            M[1][2] /= scaleFactor # Y
            tmats.append(M)
            with self.lock:
                self.count.value += 1
            conn.send("Aligning Frames")
        rets[processId] = tmats
    
    # Multiprocess function to transform and save the images to cache
    def transform(self, frames, tmats, conn):
        i = 0
        for frame in frames:
            M = tmats[i]
            image = cv2.imread(frame,1).astype(np.float32) / 255
            w, h, _ = image.shape
            image = cv2.warpPerspective(image, M, (h, w))
            h, w = image.shape[:2]
            image = image[self.maxY:h+self.minY, self.maxX:w+self.minX]
            cv2.imwrite(frame.replace("frames", "cache"),(image*255).astype(np.uint8))
            i += 1
            with self.lock:
                self.count.value += 1
            conn.send("Transforming Frames")
            
    # Multiprocess function to find the best images (ones closest to the reference frame)
    def filter(self, processId, rets, frames, conn):
        similarities = []
        ref = cv2.imread("cache/" + g.reference + ".png", cv2.IMREAD_GRAYSCALE)
        for frame in frames:
            img = cv2.imread(frame.replace("frames", "cache"), cv2.IMREAD_GRAYSCALE)
            diff = image_similarity_vectors_via_numpy("cache/" + g.reference + ".png", frame.replace("frames", "cache"))
            similarities.append((frame.replace("frames", "cache"), diff))
            with self.lock:
                self.count.value += 1
            conn.send("Calculating Similarities")
        rets[processId] = similarities

# https://github.com/petermat/image_similarity
def image_similarity_vectors_via_numpy(filepath1, filepath2):
    # source: http://www.syntacticbayleaves.com/2008/12/03/determining-image-similarity/
    # may throw: Value Error: matrices are not aligned . 
    
    image1 = Image.open(filepath1)
    image2 = Image.open(filepath2)
 
    image1 = get_thumbnail(image1, stretch_to_fit=True, size=(64,64), greyscale=True)
    image2 = get_thumbnail(image2, stretch_to_fit=True, size=(64,64), greyscale=True)
    
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(np.average(pixel_tuple))
        vectors.append(vector)
        norms.append(np.linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # ValueError: matrices are not aligned !
    res = np.dot(a / a_norm, b / b_norm)
    return res
    
def get_thumbnail(image, size=(128,128), stretch_to_fit=False, greyscale=False):
    " get a smaller version of the image - makes comparison much faster/easier"
    if not stretch_to_fit:
        image.thumbnail(size, Image.ANTIALIAS)
    else:
        image = image.resize(size); # for faster computation
    if greyscale:
        image = image.convert("L")  # Convert it to grayscale.
    return image
