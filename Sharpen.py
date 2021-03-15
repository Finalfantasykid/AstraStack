import cv2
import numpy as np
import math
import copy
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager, Lock
from pywt import swt2, iswt2
from time import sleep
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
            stackedImage = cv2.imread(stackedImage)
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
        if((g.deconvolveCircular and g.deconvolveDiameter1 > 1) or 
           (g.deconvolveLinear and g.deconvolveDiameter2 > 1) or
           (g.deconvolveCustom and isinstance(g.deconvolveFile,np.ndarray))):
            # Decompose
            (R, G, B) = cv2.split(img)
            
            futures = []
            futures.append(pool.submit(deconvolve, R, g.deconvolveDiameter1, g.deconvolveAmount1, 
                                                      g.deconvolveDiameter2, g.deconvolveAmount2, g.deconvolveAngle2,
                                                      g.deconvolveFile, g.deconvolveAmount3,
                                                      g.deconvolveCircular, g.deconvolveLinear, g.deconvolveCustom))
            futures.append(pool.submit(deconvolve, G, g.deconvolveDiameter1, g.deconvolveAmount1, 
                                                      g.deconvolveDiameter2, g.deconvolveAmount2, g.deconvolveAngle2,
                                                      g.deconvolveFile, g.deconvolveAmount3,
                                                      g.deconvolveCircular, g.deconvolveLinear, g.deconvolveCustom))
            futures.append(pool.submit(deconvolve, B, g.deconvolveDiameter1, g.deconvolveAmount1, 
                                                      g.deconvolveDiameter2, g.deconvolveAmount2, g.deconvolveAngle2,
                                                      g.deconvolveFile, g.deconvolveAmount3,
                                                      g.deconvolveCircular, g.deconvolveLinear, g.deconvolveCustom))
            
            R = futures[0].result()
            G = futures[1].result()
            B = futures[2].result()
        
            # Recompose
            img = cv2.merge([R, G, B])
            
        self.debluredImage = img
    
    # Apply brightness & color sliders
    def processColor(self):
        img = self.debluredImage
        
        # Black Level
        img = (img - (g.blackLevel/255))*(255/max(1, (255-g.blackLevel)))
        
        # Gamma
        img[img<0] = 0
        img[img>1] = 1
        img = pow(img, 1/(max(1, g.gamma)/100))
        
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
    
# Deconvolution adapted from https://github.com/opencv/opencv/blob/master/samples/python/deconvolution.py
def deconvolve(img, diameter1, strength1, diameter2, strength2, angle, psfFile, strength3, circular, linear, custom):
    # Do some initial checks
    if(not circular or diameter1 <= 1):
        diameter1 = 0
        circular = False
    if(not linear or diameter2 <= 1):
        diameter2 = 0
        linear = False
    if(not custom or not isinstance(psfFile,np.ndarray)):
        diameter3 = 0
        custom = False
    if(custom):
        diameter3 = max(psfFile.shape)
    diameter1 = int(diameter1)
    diameter2 = int(diameter2)
    
    d = max(diameter1,diameter2,diameter3)
    beforeAvg = np.average(img)
    rows, cols = img.shape
    # Resize so that the dimensions are a multiple of the diameter
    rowMod = d - ((rows+d*2) % d)
    colMod = d - ((cols+d*2) % d)
    top = d + math.floor(rowMod/2)
    bottom = d + math.ceil(rowMod/2)
    left = d + math.floor(colMod/2)
    right = d + math.ceil(colMod/2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
    img = blur_edge(img, d*2, (d + math.ceil(max(rowMod/2, colMod/2)))*2)
    
    # Start the deconvolution
    for deconvolveType in ("circular", "linear", "custom"):
        if(deconvolveType == "circular" and circular):
            psf = defocus_kernel(diameter1, diameter1*2)
            noise = 10**(-0.1*strength1)
        elif(deconvolveType == "linear" and linear):
            psf = motion_kernel(angle, diameter2, diameter2*2)
            noise = 10**(-0.1*strength2)
        elif(deconvolveType == "custom" and custom):
            psf = psfFile
            noise = 10**(-0.1*strength3)
        else:
            continue
        IMG = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        psf /= psf.sum()
        psf_pad = np.zeros_like(img)
        kh, kw = psf.shape
        psf_pad[:kh, :kw] = psf
        PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
        PSF2 = (PSF**2).sum(-1)
        iPSF = PSF / (PSF2 + noise)[...,np.newaxis]
        RES = cv2.mulSpectrums(IMG, iPSF, 0)
        img = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        img = np.roll(img, -kh//2, 0)
        img = np.roll(img, -kw//2, 1)
    
    # Crop the image to the original dimensions
    img = img[top:-bottom,left:-right]
    
    # Brightness tends to darken the image slightly, so adjust the brightness
    try:
        afterAvg = np.average(img)
        factor = beforeAvg / afterAvg
        img *= factor
    except:
        pass
    return img

def blur_edge(img, r, d):
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*r+1, 2*r+1), -1)[r:-r,r:-r]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/(d), 1.0)
    return img*w + img_blur*(1-w)

def defocus_kernel(d, sz=100):
    kern = np.zeros((sz*2, sz*2), np.uint8)
    cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA)
    kern = cv2.resize(kern, (sz, sz), interpolation=cv2.INTER_LINEAR)
    kern = np.float32(kern) / 255.0
    return kern
    
def motion_kernel(angle, d, sz=100):
    sz = max(10,sz)
    angle = np.deg2rad(angle)
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_LINEAR)
    return kern
