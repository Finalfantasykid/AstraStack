import cv2
import numpy as np
import math
import copy
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager, Lock
from pywt import swt2, iswt2
from time import sleep
from deconvolution import *
from Video import Video
from Globals import g

pool = ProcessPoolExecutor(3)

class Sharpen:

    LEVEL = 5
    
    # This is a crude estimate on how much memory will be used by the sharpening
    # Width * Height * 3Channels * 4bytes * 4coefficients * 5layers * 2(worst case from copying every coefficients layer)
    def estimateMemoryUsage(width, height):
        return width*height*3*4*4*Sharpen.LEVEL*2

    def __init__(self, stackedImage, isFile=False):
        if(isFile):
            # Single image provided
            video = Video()
            stackedImage = video.getFrame(None, stackedImage, (g.colorMode or g.guessedColorMode))
        else:
            # Use the higher bit depth version from the stacking process
            pass
        stackedImage = cv2.cvtColor(stackedImage, cv2.COLOR_BGR2RGB)
        self.h, self.w = stackedImage.shape[:2]
        self.sharpenedImage = stackedImage
        self.debluredImage = stackedImage
        self.finalImage = stackedImage
        self.calculateCoefficients(stackedImage)
        self.processAgain = False
        self.processDeblurAgain = False
        self.processColorAgain = False
        
    def run(self, processAgain=True, processDeblurAgain=False, processColorAgain=False):
        self.processAgain = processAgain
        self.processDeblurAgain = processDeblurAgain
        self.processColorAgain = processColorAgain
        while(self.processAgain or self.processDeblurAgain or self.processColorAgain):
            if(self.processAgain):
                # Process sharpening, deblur and color
                self.processAgain = False
                self.processDeblurAgain = False
                self.processColorAgain = False
                self.sharpenLayers()
                self.deblur()
                self.processColor()
            elif(self.processDeblurAgain):
                # Process deblur and color
                self.processAgain = False
                self.processDeblurAgain = False
                self.processColorAgain = False
                self.deblur()
                self.processColor()
            else:
                # Only process color
                self.processAgain = False
                self.processDeblurAgain = False
                self.processColorAgain = False
                self.processColor()
            g.ui.finishedSharpening()
        
    # Calculates the wavelet coefficents for each channel
    def calculateCoefficients(self, stackedImage):
        (R, G, B) = cv2.split(stackedImage)

        with Manager() as manager:
            num = manager.Value('i', 0)
            lock = manager.Lock()
                
            futures = []
            futures.append(pool.submit(calculateChannelCoefficients, R, 'R', num, lock))
            futures.append(pool.submit(calculateChannelCoefficients, G, 'G', num, lock))
            futures.append(pool.submit(calculateChannelCoefficients, B, 'B', num, lock))
            
            wait(futures)
        
    # Sharpens each channel
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
                
        self.sharpenedImage = cv2.merge([R, G, B])[:self.h,:self.w]
        
    def deblur(self):
        img = self.sharpenedImage
        # Deconvolve
        if((g.deconvolveCircular and g.deconvolveCircularDiameter > 1) or 
           (g.deconvolveGaussian and g.deconvolveGaussianDiameter > 1) or 
           (g.deconvolveLinear and g.deconvolveLinearDiameter > 1) or
           (g.deconvolveCustom and g.deconvolveCustomFile is not None)):
            # Decompose
            beforeAvg = np.average(img)
            (R, G, B) = cv2.split(img)
            
            gParam = {
                'circular': g.deconvolveCircular,
                'gaussian': g.deconvolveGaussian,
                'linear': g.deconvolveLinear,
                'custom': g.deconvolveCustom,
                'circularDiameter': g.deconvolveCircularDiameter,
                'circularAmount': g.deconvolveCircularAmount,
                'gaussianDiameter': g.deconvolveGaussianDiameter,
                'gaussianAmount': g.deconvolveGaussianAmount,
                'gaussianSpread': g.deconvolveGaussianSpread,
                'linearDiameter': g.deconvolveLinearDiameter,
                'linearAmount': g.deconvolveLinearAmount,
                'linearAngle': g.deconvolveLinearAngle,
                'customFile': g.deconvolveCustomFile,
                'customAmount': g.deconvolveCustomAmount
            }
            
            futures = []
            futures.append(pool.submit(deconvolve, R, gParam))
            futures.append(pool.submit(deconvolve, G, gParam))
            futures.append(pool.submit(deconvolve, B, gParam))
            
            R = futures[0].result()
            G = futures[1].result()
            B = futures[2].result()
        
            # Recompose
            img = cv2.merge([R, G, B])
            # Brightness tends to darken the image slightly, so adjust the brightness
            try:
                afterAvg = np.average(img)
                factor = beforeAvg / afterAvg
                img *= factor
            except:
                pass
           
        self.debluredImage = img
    
    # Apply brightness & color sliders
    def processColor(self):
        img = self.debluredImage
        # Black Level
        img = (img - (g.blackLevel/255))*(255/max(1, (255-g.blackLevel)))

        # Decompose
        (R, G, B) = cv2.split(img)
        
        # Red Adjust
        R *= (g.redAdjust/100)*g.value/100
        
        # Green Adjust
        G *= (g.greenAdjust/100)*g.value/100
        
        # Blue Adjust
        B *= (g.blueAdjust/100)*g.value/100
        
        # Recompose
        img = cv2.merge([R, G, B])
        img[img<0] = 0
        img[img>1] = 1
        
        # Saturation
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        (H, V, S) = cv2.split(img)
        S *= g.saturation/100
        img = cv2.merge([H, V, S])
        
        img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
        
        # Gamma
        img[img<0] = 0
        img[img>1] = 1
        img = pow(img, 1/(max(1, g.gamma)/100))
        
        # Clip at 0 and 255
        img *= 255
        img[img>255] = 255
        img[img<0] = 0
        
        self.finalImage = img
        
# Calculates the wavelet coefficents for the specified channel
def calculateChannelCoefficients(C, channel, num, lock):
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

    # Compute coefficients
    g.coeffs = list(swt2(C, 'haar', level=Sharpen.LEVEL, trim_approx=True))
    g.channel = channel
    
    # Make sure that the other processes have completed as well
    with lock:
        num.value += 1
    while(num.value < 3):
        sleep(0.01)
    
# Reconstructs the wavelet using varying intensities of coefficients, 
# as well as unsharp mask for additional sharpening
# and gaussian denoise
def sharpenChannelLayers(params):
    c = g.coeffs
    # Go through each wavelet layer and apply sharpening
    cCopy = []
    for i in range(1, len(c)):
        level = (len(c) - i - 1)
        # Copy the layer if a change is made to the coefficients
        if(params['level'][level] and (params['radius'][level] > 0 or 
                                       params['sharpen'][level] > 0 or 
                                       params['denoise'][level] > 0)):
            cCopy.append(copy.deepcopy(c[i]))
        else:
            cCopy.append(None)
            
        # Process Layers
        if(params['level'][level]):
            # Apply Unsharp Mask
            if(params['radius'][level] > 0):
                unsharp(c[i][0], params['radius'][level], 2)
                unsharp(c[i][1], params['radius'][level], 2)
                unsharp(c[i][2], params['radius'][level], 2)
            # Multiply the layer to increase intensity
            if(params['sharpen'][level] > 0):
                factor = (100 - 10*level)
                cv2.add(c[i][0], c[i][0]*params['sharpen'][level]*factor, c[i][0])
                cv2.add(c[i][1], c[i][1]*params['sharpen'][level]*factor, c[i][1])
                cv2.add(c[i][2], c[i][2]*params['sharpen'][level]*factor, c[i][2])
            # Denoise
            if(params['denoise'][level] > 0):
                unsharp(c[i][0], params['denoise'][level], -1)
                unsharp(c[i][1], params['denoise'][level], -1)
                unsharp(c[i][2], params['denoise'][level], -1)
    
    # Reconstruction
    padding = 2**(Sharpen.LEVEL)
    img=iswt2(c, 'haar')
    
    # Reset coefficients
    for i in range(1, len(c)):
        if(cCopy[i-1] is not None):
            c[i] = cCopy[i-1]
    
    # Prepare image for saving
    img = img[padding:,padding:]
    
    return (g.channel, img)
    
# Applies an unsharp mask to the specified image
def unsharp(image, radius, strength):
    blur = cv2.GaussianBlur(image, (0,0), radius)
    sharp = cv2.addWeighted(image, 1+strength, blur, -strength, 0, image)

