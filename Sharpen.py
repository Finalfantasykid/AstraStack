import cv2
import numpy as np
import math
import copy
from concurrent.futures import ProcessPoolExecutor
from pywt import swt2, iswt2

from Globals import g

pool = ProcessPoolExecutor(3)

class Sharpen:

    LEVEL = 5
    
    # This is a crude estimate on how much memory will be used by the sharpening
    # Width * Height * 3Channels * 4bytes * 4coefficients * 5layers * 2(while deep copying)
    def estimateMemoryUsage(width, height):
        return width*height*3*4*4*Sharpen.LEVEL*2

    def __init__(self, stackedImage, isFile=False):
        if(isFile):
            # Single image provided
            stackedImage = cv2.imread(stackedImage)
        else:
            # Use the higher bit depth version from the stacking process
            pass
        self.h, self.w = stackedImage.shape[:2]
        self.calculateCoefficients(stackedImage)
        self.processAgain = False
        
    def run(self):
        self.processAgain = True
        while(self.processAgain):
            self.processAgain = False
            self.sharpenLayers()
            g.ui.finishedSharpening()
        
    def calculateCoefficients(self, stackedImage):
        (B, G, R) = cv2.split(stackedImage)
        
        futureR = pool.submit(calculateChannelCoefficients, R, 'R')
        futureG = pool.submit(calculateChannelCoefficients, G, 'G')
        futureB = pool.submit(calculateChannelCoefficients, B, 'B')
        
        futureR.result()
        futureG.result()
        futureB.result()
        
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

        futures = []

        futures.append(pool.submit(sharpenChannelLayers, gParam))
        futures.append(pool.submit(sharpenChannelLayers, gParam))
        futures.append(pool.submit(sharpenChannelLayers, gParam))
        
        for future in futures:
            result = future.result()
            if(result[0] == 'R'):
                R = result[1]
            elif(result[0] == 'G'):
                G = result[1]
            elif(result[0] == 'B'):
                B = result[1]
        
        cv2.imwrite(g.tmp + "sharpened.png", cv2.merge([B, G, R])[:self.h,:self.w])
        
def calculateChannelCoefficients(C, channel):
    # Pad the image so that there is a border large enough so that edge artifacts don't occur
    
    padding = 2**(Sharpen.LEVEL)
    C = cv2.copyMakeBorder(C, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    
    # Pad so that dimensions are multiples of 2**level
    h, w = C.shape[:2]
    
    hR = (h % 2**Sharpen.LEVEL)
    wR = (w % 2**Sharpen.LEVEL)

    C = cv2.copyMakeBorder(C, 0, (2**Sharpen.LEVEL - hR), 0, (2**Sharpen.LEVEL - wR), cv2.BORDER_REFLECT)
    
    C =  np.float32(C)
    C /= 255

    # compute coefficients
    g.coeffs = list(swt2(C, 'haar', level=Sharpen.LEVEL, trim_approx=True))
    g.channel = channel
    
def sharpenChannelLayers(params):
    c = copy.deepcopy(g.coeffs)
    # Go through each wavelet layer and apply sharpening
    for i in range(1, len(c)):
        level = (len(c) - i - 1)
        if(params['level'][level]):
            if(params['radius'][level] > 0):
                unsharp(c[i][0], params['radius'][level], 1)
                unsharp(c[i][1], params['radius'][level], 1)
                unsharp(c[i][2], params['radius'][level], 1)
            if(params['sharpen'][level] > 0):
                cv2.add(c[i][0], c[i][0]*params['sharpen'][level]*50, c[i][0])
                cv2.add(c[i][1], c[i][1]*params['sharpen'][level]*50, c[i][1])
                cv2.add(c[i][2], c[i][2]*params['sharpen'][level]*50, c[i][2])
            if(params['denoise'][level] > 0):
                unsharp(c[i][0], params['denoise'][level]*10, -1)
                unsharp(c[i][1], params['denoise'][level]*10, -1)
                unsharp(c[i][2], params['denoise'][level]*10, -1)
    
    # reconstruction
    padding = 2**(Sharpen.LEVEL)
    img=iswt2(c, 'haar')
    img = img[padding:,padding:]
    img *= 255
    img[img>255] = 255
    img[img<0] = 0
    return (g.channel, np.uint8(img))
    
def unsharp(image, radius, strength):
    blur = cv2.GaussianBlur(image, (0,0), radius)
    sharp = cv2.addWeighted(image, 1+strength, blur, -strength, 0, image)
