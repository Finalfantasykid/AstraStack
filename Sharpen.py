import cv2
import numpy as np
import math
import multiprocessing
from pywt import swt2, iswt2

from Globals import g

class Sharpen:

    LEVEL = 5
    
    # This is a crude estimate on how much memory will be used by the sharpening
    # Width * Height * 3Channels * 4bytes * 4coefficients * 5layers * 3processes
    def estimateMemoryUsage(width, height):
        return width*height*3*4*4*Sharpen.LEVEL*3

    def __init__(self, stackedImage, isFile=False):
        if(isFile):
            # Single image provided
            stackedImage = cv2.imread(stackedImage)
        else:
            # Use the higher bit depth version from the stacking process
            pass
        self.h, self.w = stackedImage.shape[:2]
        mR = multiprocessing.Manager()
        mG = multiprocessing.Manager()
        mB = multiprocessing.Manager()
        self.R = mR.dict()
        self.G = mG.dict()
        self.B = mB.dict()
        self.calculateCoefficients(stackedImage)
        self.processAgain = False
        
    def run(self):
        self.processAgain = True
        while(self.processAgain):
            self.processAgain = False
            self.sharpenLayers()
            g.ui.finishedSharpening()
        
    def calculateCoefficients(self, stackedImage):
        (self.B['img'], self.G['img'], self.R['img']) = cv2.split(stackedImage)
        
        futureR = g.pool.submit(calculateChannelCoefficients, self.R)
        futureG = g.pool.submit(calculateChannelCoefficients, self.G)
        futureB = g.pool.submit(calculateChannelCoefficients, self.B)
        
        futureR.result()
        futureG.result()
        futureB.result()
        
        # Clean up memory since we don't need this anymore
        del self.R['img']
        del self.G['img']
        del self.B['img']
        del stackedImage
        
    def sharpenLayers(self):
        gParam = {
            'level' : [g.level1, 
                       g.level2,
                       g.level3,
                       g.level4,
                       g.level5],
            'sharpen': [g.sharpen1,
                        g.sharpen2,
                        g.sharpen3,
                        g.sharpen4,
                        g.sharpen5],
            'radius': [g.radius1,
                       g.radius2,
                       g.radius3,
                       g.radius4,
                       g.radius5],
            'denoise': [g.denoise1,
                        g.denoise2,
                        g.denoise3,
                        g.denoise4,
                        g.denoise5]
        }

        futureR = g.pool.submit(sharpenChannelLayers, self.R, gParam)
        futureG = g.pool.submit(sharpenChannelLayers, self.G, gParam)
        futureB = g.pool.submit(sharpenChannelLayers, self.B, gParam)
        
        R = futureR.result()
        G = futureG.result()
        B = futureB.result()
        
        cv2.imwrite(g.tmp + "sharpened.png", cv2.merge([B, G, R])[:self.h,:self.w])
        
def calculateChannelCoefficients(C):
    # Pad the image so that there is a border large enough so that edge artifacts don't occur
    padding = 2**(Sharpen.LEVEL)
    C['img'] = cv2.copyMakeBorder(C['img'], padding, padding, padding, padding, cv2.BORDER_REFLECT)
    
    # Pad so that dimensions are multiples of 2**level
    h, w = C['img'].shape[:2]
    
    hR = (h % 2**Sharpen.LEVEL)
    wR = (w % 2**Sharpen.LEVEL)

    C['img'] = cv2.copyMakeBorder(C['img'], 0, (2**Sharpen.LEVEL - hR), 0, (2**Sharpen.LEVEL - wR), cv2.BORDER_REFLECT)
    
    C['img'] =  np.float32(C['img'])
    C['img'] /= 255

    # compute coefficients
    C['c'] = list(swt2(C['img'], 'haar', level=Sharpen.LEVEL, trim_approx=True))
    
def sharpenChannelLayers(C, g):
    c = C['c']
    # Go through each wavelet layer and apply sharpening
    for i in range(1, len(c)):
        level = (len(c) - i - 1)
        if(g['level'][level]):
            if(g['radius'][level] > 0):
                unsharp(c[i][0], g['radius'][level], 1)
                unsharp(c[i][1], g['radius'][level], 1)
                unsharp(c[i][2], g['radius'][level], 1)
            if(g['sharpen'][level] > 0):
                cv2.add(c[i][0], c[i][0]*g['sharpen'][level]*50, c[i][0])
                cv2.add(c[i][1], c[i][1]*g['sharpen'][level]*50, c[i][1])
                cv2.add(c[i][2], c[i][2]*g['sharpen'][level]*50, c[i][2])
            if(g['denoise'][level] > 0):
                unsharp(c[i][0], g['denoise'][level]*10, -1)
                unsharp(c[i][1], g['denoise'][level]*10, -1)
                unsharp(c[i][2], g['denoise'][level]*10, -1)
    
    # reconstruction
    padding = 2**(Sharpen.LEVEL)
    img=iswt2(c, 'haar')
    img = img[padding:,padding:]
    img *= 255
    img[img>255] = 255
    img[img<0] = 0
    return np.uint8(img)
    
def unsharp(image, radius, strength):
    blur = cv2.GaussianBlur(image, (0,0), radius)
    sharp = cv2.addWeighted(image, 1+strength, blur, -strength, 0, image)
