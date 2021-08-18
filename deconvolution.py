import cv2
import numpy as np
import math

# Deconvolution adapted from https://github.com/opencv/opencv/blob/master/samples/python/deconvolution.py
def deconvolve(img, gCopy):
    # Do some initial checks
    if(not gCopy.deconvolveCircular or gCopy.deconvolveCircularDiameter <= 1):
        gCopy.deconvolveCircularDiameter = 0
        gCopy.deconvolveCircular = False
    if(not gCopy.deconvolveGaussian or gCopy.deconvolveGaussianDiameter <= 1):
        gCopy.deconvolveGaussianDiameter = 0
        gCopy.deconvolveGaussian = False
    if(not gCopy.deconvolveLinear or gCopy.deconvolveLinearDiameter <= 1):
        gCopy.deconvolveLinearDiameter = 0
        gCopy.deconvolveLinear = False
    if(not gCopy.deconvolveCustom or gCopy.deconvolveCustomFile is None):
        gCopy.deconvolveCustomDiameter = 0
        gCopy.deconvolveCustom = False
    if(gCopy.deconvolveCustom):
        gCopy.deconvolveCustomDiameter = max(gCopy.deconvolveCustomFile.shape)
    gCopy.deconvolveCircularDiameter = int(gCopy.deconvolveCircularDiameter)
    gCopy.deconvolveGaussianDiameter = int(gCopy.deconvolveGaussianDiameter)
    gCopy.deconvolveLinearDiameter = int(gCopy.deconvolveLinearDiameter)
    
    d = max(gCopy.deconvolveCircularDiameter + gCopy.deconvolveLinearDiameter - 1,
            gCopy.deconvolveGaussianDiameter + gCopy.deconvolveLinearDiameter - 1,
            gCopy.deconvolveLinearDiameter,
            gCopy.deconvolveCustomDiameter)
    if(gCopy.deconvolveGaussianDiameter > 1):
        d = max(d, 10)
    d = max(d, 1) # Make sure 'd' is always at least 1
    rows, cols = img.shape
    # Resize so that the dimensions are a multiple of the diameter
    rowMod = d - ((rows+d*2) % d)
    colMod = d - ((cols+d*2) % d)
    top = d + math.floor(rowMod/2)
    bottom = d + math.ceil(rowMod/2)
    left = d + math.floor(colMod/2)
    right = d + math.ceil(colMod/2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
    img = blur_edge(img, d, d)
    
    # Start the deconvolution
    for deconvolveType in ("circular", "gaussian", "linear", "custom"):
        if(deconvolveType == "circular" and gCopy.deconvolveCircular):
            # Circular
            psf = defocus_kernel(gCopy.deconvolveCircularDiameter, (gCopy.deconvolveCircularDiameter + gCopy.deconvolveLinearDiameter - 1)*2) 
            noise = 10**(-0.1*gCopy.deconvolveCircularAmount)
        elif(deconvolveType == "gaussian" and gCopy.deconvolveGaussian):
            # Gaussian
            psf = gaussian_kernel(gCopy.deconvolveGaussianDiameter, gCopy.deconvolveGaussianSpread, max(10, (gCopy.deconvolveGaussianDiameter + gCopy.deconvolveLinearDiameter - 1)*2))
            noise = 10**(-0.1*gCopy.deconvolveGaussianAmount)
        elif(deconvolveType == "custom" and gCopy.deconvolveCustom):
            # Custom
            psf = gCopy.deconvolveCustomFile
            noise = 10**(-0.1*gCopy.deconvolveCustomAmount)
        elif(deconvolveType == "linear" and gCopy.deconvolveLinear and not gCopy.deconvolveCircular and not gCopy.deconvolveGaussian):
            # Linear Only
            psf = motion_kernel(gCopy.deconvolveLinearAngle, gCopy.deconvolveLinearDiameter, (gCopy.deconvolveLinearDiameter*2 - 1)*2)
            noise = 10**(-0.1*gCopy.deconvolveLinearAmount)
        else:
            continue
        if((deconvolveType == "circular" or deconvolveType == "gaussian") and gCopy.deconvolveLinear):
            # Stretch&Rotate the Gaussian/Circular
            psf = stretchPSF(psf, (gCopy.deconvolveLinearDiameter/max(gCopy.deconvolveGaussianDiameter, gCopy.deconvolveCircularDiameter)), gCopy.deconvolveLinearAngle)
        kh, kw = psf.shape   
        IMG = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        psf /= psf.sum()
        psf_pad = np.zeros_like(img)
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

    return img

# Blurs the edge of the image to reduce boundary artifacts
def blur_edge(img, r, d):
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*r+1, 2*r+1), -1)[r:-r,r:-r]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/(d), 1.0)
    return img*w + img_blur*(1-w)

# Returns a circular psf
def defocus_kernel(d, sz=100):
    kern = np.zeros((sz*2, sz*2), np.float32)
    cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA)
    kern = cv2.resize(kern, (sz, sz), interpolation=cv2.INTER_LINEAR)
    kern = np.float32(kern) / 255.0
    return kern
    
# Returns a gaussian/moffat/lorentzian psf
def gaussian_kernel(d, spread=10, sz=100):
    kern = np.zeros((sz, sz), np.float32)
    if(spread > 10):
        # Gaussian
        d1 = d*1.025
        kern[math.ceil(sz/2)][math.ceil(sz/2)] = 1
        kern = cv2.GaussianBlur(kern, (0,0), (d1/4))
    else:
        # Moffat
        d1 = d*(1+(1/(spread+1)))
        d1 = d1/(10/spread)
        x0 = sz/2
        y0 = sz/2
        for y, row in enumerate(kern):
            for x, col in enumerate(row):
                kern[y][x] = (1 + ((x-x0)**2 + (y-y0)**2)/d1**2)**(-spread)
    kern = blackenEdge(kern)
    return kern
    
# Returns a linear motion psf
def motion_kernel(angle, d, sz=100):
    angle = np.deg2rad(angle)
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_LINEAR)
    return kern
    
# Stretches and rotates a given psf
def stretchPSF(psf, d, angle):
    h, w  = psf.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1)
    T = np.identity(3) # Scale Matrix
    T[0][0] = 1 + d
    T[0][2] -= (w/2)*(T[0][0]-1)
    M = M.dot(T) # Apply scale to Transformation
    psf = cv2.warpAffine(psf, M, (w, h), flags=cv2.INTER_LINEAR)
    return blackenEdge(psf)
    
# Adjusts the brightness of the psf so that the edge is always black
def blackenEdge(kern):
    h, w  = kern.shape[:2]
    edgeMax = max(kern[0].max(), 
                  kern[h-1].max(),
                  kern[:,0].max(),
                  kern[:,w-1].max())
    kern -= edgeMax
    kern[kern<0] = 0
    return kern
