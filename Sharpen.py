import cv2
import numpy as np
import math
from pywt import swt2, iswt2
from multiprocessing import Manager, Process

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
        
    def calculateChannelCoefficients(self, img, level, rets, channel):
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
        rets[channel] = list(swt2(img, 'haar', level=level))

    def sharpenChannelLayers(self, c, rets, channel):
        if(g.level3):
            if(g.sharpen3 > 0):
                self.unsharp(c[0][1][0], g.radius3, g.sharpen3*100)
                self.unsharp(c[0][1][1], g.radius3, g.sharpen3*100)
                self.unsharp(c[0][1][2], g.radius3, g.sharpen3*100)
            if(g.denoise3 > 0):
                self.unsharp(c[0][1][0], g.denoise3, -1)
                self.unsharp(c[0][1][1], g.denoise3, -1)
                self.unsharp(c[0][1][2], g.denoise3, -1)
        
        if(g.level2):
            if(g.sharpen2 > 0):
                self.unsharp(c[1][1][0], g.radius2, g.sharpen2*100)
                self.unsharp(c[1][1][1], g.radius2, g.sharpen2*100)
                self.unsharp(c[1][1][2], g.radius2, g.sharpen2*100)
            if(g.denoise2 > 0):
                self.unsharp(c[1][1][0], g.denoise2, -1)
                self.unsharp(c[1][1][1], g.denoise2, -1)
                self.unsharp(c[1][1][2], g.denoise2, -1)
        
        if(g.level1):
            if(g.sharpen1 > 0):
                self.unsharp(c[2][1][0], g.radius1, g.sharpen1*100)
                self.unsharp(c[2][1][1], g.radius1, g.sharpen1*100)
                self.unsharp(c[2][1][2], g.radius1, g.sharpen1*100)

            if(g.denoise1 > 0):
                self.unsharp(c[2][1][0], g.denoise1, -1)
                self.unsharp(c[2][1][1], g.denoise1, -1)
                self.unsharp(c[2][1][2], g.denoise1, -1)
        
        # reconstruction
        img=iswt2(c, 'haar');
        img *= 255;
        img[img>255] = 255
        img[img<0] = 0
        rets[channel] = np.uint8(img)
        
    def calculateCoefficients(self):
        (imgB, imgG, imgR) = cv2.split(self.stackedImage)
        manager = Manager()
        rets = manager.dict()
        threads = []
        threads.append(Process(target=self.calculateChannelCoefficients, args=(imgR, 3, rets, 'R', )))
        threads.append(Process(target=self.calculateChannelCoefficients, args=(imgG, 3, rets, 'G', )))
        threads.append(Process(target=self.calculateChannelCoefficients, args=(imgB, 3, rets, 'B', )))
        
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        
        self.cR = rets['R']
        self.cG = rets['G']
        self.cB = rets['B']
        
    def sharpenLayers(self):
        manager = Manager()
        rets = manager.dict()
        threads = []
        threads.append(Process(target=self.sharpenChannelLayers, args=(self.cR, rets, 'R', )))
        threads.append(Process(target=self.sharpenChannelLayers, args=(self.cG, rets, 'G', )))
        threads.append(Process(target=self.sharpenChannelLayers, args=(self.cB, rets, 'B', )))
        
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        
        h, w = self.finalImage.shape[:2]
        self.finalImage = cv2.merge([rets['B'], rets['G'], rets['R']])
        self.finalImage = self.finalImage[:h,:w]
        cv2.imwrite("sharpened.png", self.finalImage)
    
    def unsharp(self, image, radius, strength):
        kSize = max(3, math.ceil(radius*3) + (math.ceil(radius*3)+1) % 2) # kernel size should be at least 3 times the radius
        blur = cv2.GaussianBlur(image, (kSize,kSize), radius)
        sharp = cv2.addWeighted(image, 1+strength, blur, -strength, 0, image)
        return sharp
