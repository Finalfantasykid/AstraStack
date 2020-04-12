import cv2
import numpy as np
import math
from pywt import swt2, iswt2

from Globals import g

class Sharpen:

    def __init__(self, stackedImage):
        self.stackedImage = cv2.imread(stackedImage)
        self.finalImage = None
        self.cR = None
        self.cG = None
        self.cB = None
        self.calculateCoefficients()
        self.processAgain = False
        
    def run(self):
        self.finalImage = self.stackedImage
        self.processAgain = True
        while(self.processAgain):
            self.processAgain = False
            self.sharpenLayers()
            g.ui.finishedSharpening()
        
    def calculateCoefficients(self):
        (imgB, imgG, imgR) = cv2.split(self.stackedImage)
        
        futureR = g.pool.submit(calculateChannelCoefficients, imgR, 5)
        futureG = g.pool.submit(calculateChannelCoefficients, imgG, 5)
        futureB = g.pool.submit(calculateChannelCoefficients, imgB, 5)
        
        self.cR = futureR.result()
        self.cG = futureG.result()
        self.cB = futureB.result()
        
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
                  
        futureR = g.pool.submit(sharpenChannelLayers, self.cR, gParam)
        futureG = g.pool.submit(sharpenChannelLayers, self.cG, gParam)
        futureB = g.pool.submit(sharpenChannelLayers, self.cB, gParam)
        
        R = futureR.result()
        G = futureG.result()
        B = futureB.result()
        
        h, w = self.finalImage.shape[:2]
        self.finalImage = cv2.merge([B, G, R])
        self.finalImage = self.finalImage[:h,:w]
        cv2.imwrite(g.tmp + "sharpened.png", self.finalImage)
        
def calculateChannelCoefficients(img, level):
    # Crop so that dimensions are multiples of 2**level
    h, w = img.shape[:2]
    
    hR = (h % 2**level)
    wR = (w % 2**level)

    img = cv2.copyMakeBorder(img, 0, (2**level - hR), 0, (2**level - wR), cv2.BORDER_REFLECT)

    img =  np.float32(img) 
    img /= 255;
    
    cv2.imwrite("test.png", (img*255).astype(np.uint8))
    
    # compute coefficients
    return list(swt2(img, 'haar', level=level))
    
def sharpenChannelLayers(c, g):
    # Go through each wavelet layer and apply sharpening
    for i in range(0, len(c)):
        level = (len(c) - i - 1)
        if(g['level'][level]):
            if(g['sharpen'][level] > 0):
                unsharp(c[i][1][0], g['radius'][level], g['sharpen'][level]*100)
                unsharp(c[i][1][1], g['radius'][level], g['sharpen'][level]*100)
                unsharp(c[i][1][2], g['radius'][level], g['sharpen'][level]*100)
            if(g['denoise'][level] > 0):
                unsharp(c[i][1][0], g['denoise'][level], -1)
                unsharp(c[i][1][1], g['denoise'][level], -1)
                unsharp(c[i][1][2], g['denoise'][level], -1)
    
    # reconstruction
    img=iswt2(c, 'haar');
    img *= 255;
    img[img>255] = 255
    img[img<0] = 0
    return np.uint8(img)
    
def unsharp(image, radius, strength):
    kSize = max(3, math.ceil(radius*3) + (math.ceil(radius*3)+1) % 2) # kernel size should be at least 3 times the radius
    blur = cv2.GaussianBlur(image, (kSize,kSize), radius)
    sharp = cv2.addWeighted(image, 1+strength, blur, -strength, 0, image)
    return sharp
