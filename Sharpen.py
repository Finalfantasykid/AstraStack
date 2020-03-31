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
        
        futureR = g.pool.submit(calculateChannelCoefficients, imgR, 3)
        futureG = g.pool.submit(calculateChannelCoefficients, imgG, 3)
        futureB = g.pool.submit(calculateChannelCoefficients, imgB, 3)
        
        self.cR = futureR.result()
        self.cG = futureG.result()
        self.cB = futureB.result()
        
    def sharpenLayers(self):
        gParam = {'level1': g.level1,
                  'level2': g.level2,
                  'level3': g.level3,
                  'sharpen1': g.sharpen1,
                  'sharpen2': g.sharpen2,
                  'sharpen3': g.sharpen3,
                  'radius1': g.radius1,
                  'radius2': g.radius2,
                  'radius3': g.radius3,
                  'denoise1': g.denoise1,
                  'denoise2': g.denoise2,
                  'denoise3': g.denoise3}
                  
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
    
    tmp = np.zeros((h+(2**level - hR), w+(2**level - wR)))
    tmp[:h,:w] = img
    img = tmp

    img =  np.float32(img)   
    img /= 255;
    
    # compute coefficients
    return list(swt2(img, 'haar', level=level))
    
def sharpenChannelLayers(c, g):
    if(g['level3']):
        if(g['sharpen3'] > 0):
            unsharp(c[0][1][0], g['radius3'], g['sharpen3']*100)
            unsharp(c[0][1][1], g['radius3'], g['sharpen3']*100)
            unsharp(c[0][1][2], g['radius3'], g['sharpen3']*100)
        if(g['denoise3'] > 0):
            unsharp(c[0][1][0], g['denoise3'], -1)
            unsharp(c[0][1][1], g['denoise3'], -1)
            unsharp(c[0][1][2], g['denoise3'], -1)
    
    if(g['level2']):
        if(g['sharpen2'] > 0):
            unsharp(c[1][1][0], g['radius2'], g['sharpen2']*100)
            unsharp(c[1][1][1], g['radius2'], g['sharpen2']*100)
            unsharp(c[1][1][2], g['radius2'], g['sharpen2']*100)
        if(g['denoise2'] > 0):
            unsharp(c[1][1][0], g['denoise2'], -1)
            unsharp(c[1][1][1], g['denoise2'], -1)
            unsharp(c[1][1][2], g['denoise2'], -1)
    
    if(g['level1']):
        if(g['sharpen1'] > 0):
            unsharp(c[2][1][0], g['radius1'], g['sharpen1']*100)
            unsharp(c[2][1][1], g['radius1'], g['sharpen1']*100)
            unsharp(c[2][1][2], g['radius1'], g['sharpen1']*100)

        if(g['denoise1'] > 0):
            unsharp(c[2][1][0], g['denoise1'], -1)
            unsharp(c[2][1][1], g['denoise1'], -1)
            unsharp(c[2][1][2], g['denoise1'], -1)
    
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
