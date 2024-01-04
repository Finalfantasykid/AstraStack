from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
pywt = lazy("pywt")
import math
import copy
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager, Lock
from time import sleep
from deconvolution import *
from Video import Video
from Globals import *

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
            stackedImage = video.getFrame(None, stackedImage, g.actualColor())
        else:
            # Use the higher bit depth version from the stacking process
            pass
        self.stackedImage = cv2.cvtColor(stackedImage, cv2.COLOR_BGR2RGB)
        self.h, self.w = stackedImage.shape[:2]
        self.mask = None
        self.sharpenedImage = self.stackedImage
        self.debluredImage = self.stackedImage
        self.deringedImage = self.stackedImage
        self.finalImage = self.stackedImage
        self.calculateCoefficients(self.stackedImage)
        self.processAgain = False
        self.processDeblurAgain = False
        self.processDeringAgain = False
        self.processColorAgain = False
        
    def run(self, processAgain=True, processDeblurAgain=False, processDeringAgain=False, processColorAgain=False):
        self.processAgain = processAgain
        self.processDeblurAgain = processDeblurAgain
        self.processDeringAgain = processDeringAgain
        self.processColorAgain = processColorAgain
        while(self.processAgain or self.processDeblurAgain or self.processDeringAgain or self.processColorAgain):
            if(self.processAgain):
                # Process sharpening, deblur and color
                self.processAgain = False
                self.processDeblurAgain = False
                self.processDeringAgain = False
                self.processColorAgain = False
                self.sharpenLayers()
                self.deblur()
                self.dering()
                self.processColor()
            elif(self.processDeblurAgain):
                # Process deblur and color
                self.processAgain = False
                self.processDeblurAgain = False
                self.processDeringAgain = False
                self.processColorAgain = False
                self.deblur()
                self.dering()
                self.processColor()
            elif(self.processDeringAgain):
                # Only process color
                self.processAgain = False
                self.processDeblurAgain = False
                self.processDeringAgain = False
                self.processColorAgain = False
                self.dering()
                self.processColor()
            else:
                # Only process color
                self.processAgain = False
                self.processDeblurAgain = False
                self.processDeringAgain = False
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
        gCopy = cloneGlobals()

        futures = []
        futures.append(pool.submit(sharpenChannelLayers, gCopy))
        futures.append(pool.submit(sharpenChannelLayers, gCopy))
        futures.append(pool.submit(sharpenChannelLayers, gCopy))
        
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

            gCopy = cloneGlobals()
            
            futures = []
            futures.append(pool.submit(deconvolve, R, gCopy))
            futures.append(pool.submit(deconvolve, G, gCopy))
            futures.append(pool.submit(deconvolve, B, gCopy))
            
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
        img[img<0] = 0
        self.debluredImage = img
        
    # Apply dering/star mask to sharpened image
    def dering(self):
        if((g.showAdaptive and g.deringAdaptive > 0) or 
           (g.showDark and g.deringDark > 0) or 
           (g.showBright and g.deringBright > 0)):
            gray = cv2.cvtColor(self.stackedImage, cv2.COLOR_RGB2GRAY)
            # Calculate Dark & Bright thresholds and merge them
            threshAdaptive = np.zeros((self.h, self.w), dtype = "uint8")
            threshDark = np.zeros((self.h, self.w), dtype = "uint8")
            threshBright = np.zeros((self.h, self.w), dtype = "uint8")
            if(g.showAdaptive and g.deringAdaptive > 0):
                threshAdaptive = cv2.adaptiveThreshold(gray.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 30-g.deringAdaptive)
            if(g.showDark and g.deringDark > 0):
                ret,threshDark = cv2.threshold(gray, (g.deringDark*2)-1, 255, cv2.THRESH_BINARY_INV)
            if(g.showBright and g.deringBright > 0):
                ret,threshBright = cv2.threshold(gray, 255-(g.deringBright*2), 255, cv2.THRESH_BINARY)
            thresh = np.maximum(np.maximum(threshAdaptive, threshDark), threshBright)
            
            # Dialate the threshold
            if(g.deringSize != 0):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(g.deringSize)*2+1,abs(g.deringSize)*2+1))
                if(g.deringSize > 0):
                    thresh = cv2.dilate(thresh, kernel)
                else:
                    thresh = cv2.erode(thresh, kernel)
            
            # Blur the thresholds
            if(g.deringBlend > 0):
                thresh = cv2.GaussianBlur(thresh, (0,0), g.deringBlend)
            
            self.thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            mask = self.thresh.astype(np.float32)/255
            inv_mask = 1 - mask
            orig = self.stackedImage.astype(np.float32)/255 * mask
            processed = self.debluredImage * inv_mask
            self.deringedImage = orig + processed
        else:
            self.thresh = cv2.cvtColor(np.zeros((self.h, self.w), dtype = "uint8"), cv2.COLOR_GRAY2RGB)
            self.deringedImage = self.debluredImage
    
    # Apply brightness & color sliders
    def processColor(self):
        def clamp(img, low, high):
            img[img<low] = low
            img[img>high] = high
            return img
            
        def colorAdjust(C, adjust):
            C *= (adjust/100)*(g.value/100)
            return C
            
        img = self.deringedImage
        # Black Level
        img = (img - (g.blackLevel/255))*(255/max(1, (255-g.blackLevel)))

        # Decompose
        (R, G, B) = cv2.split(img)
        
        # Color Adjust
        R = colorAdjust(R, g.redAdjust)
        G = colorAdjust(G, g.greenAdjust)
        B = colorAdjust(B, g.blueAdjust)
        
        # Recompose
        img = cv2.merge([R, G, B])
        img = clamp(img, 0, 1)
        
        # Saturation
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        (H, V, S) = cv2.split(img)
        S *= g.saturation/100
        img = cv2.merge([H, V, S])
        
        img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
        
        # Gamma
        img = clamp(img, 0, 1)
        img = pow(img, 1/(max(1, g.gamma)/100))
        
        # Clip at 0 and 255
        img *= 255
        img = clamp(img, 0, 255)
        
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
    g.coeffs = list(pywt.swt2(C, 'haar', level=Sharpen.LEVEL, trim_approx=True))
    g.channel = channel
    
    # Make sure that the other processes have completed as well
    with lock:
        num.value += 1
    while(num.value < 3):
        sleep(0.01)
    
# Reconstructs the wavelet using varying intensities of coefficients, 
# as well as unsharp mask for additional sharpening
# and gaussian denoise
def sharpenChannelLayers(gCopy):
    c = g.coeffs
    # Go through each wavelet layer and apply sharpening
    cCopy = []
    for i in range(1, len(c)):
        level = (len(c) - i - 1)
        # Copy the layer if a change is made to the coefficients
        if(gCopy.level[level] and (gCopy.radius[level] > 0 or 
                                   gCopy.sharpen[level] > 0 or 
                                   gCopy.denoise[level] > 0)):
            cCopy.append(copy.deepcopy(c[i]))
        else:
            cCopy.append(None)
            
        # Process Layers
        if(gCopy.level[level]):
            for ci in c[i]:
                # Apply Unsharp Mask
                if(gCopy.radius[level] > 0):
                    unsharp(ci, gCopy.radius[level], 2)
                # Multiply the layer to increase intensity
                if(gCopy.sharpen[level] > 0):
                    factor = (100 - 10*level)
                    cv2.add(ci, ci*gCopy.sharpen[level]*factor, ci)
                # Denoise
                if(gCopy.denoise[level] > 0):
                    unsharp(ci, gCopy.denoise[level], -1)
    
    # Reconstruction
    padding = 2**(Sharpen.LEVEL)
    img=pywt.iswt2(c, 'haar')
    
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

