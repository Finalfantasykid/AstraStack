import cv2
import numpy as np
import math

# Deconvolution adapted from https://github.com/opencv/opencv/blob/master/samples/python/deconvolution.py
def deconvolve(img, params):
    # Do some initial checks
    if(not params['circular'] or params['circularDiameter'] <= 1):
        params['circularDiameter'] = 0
        params['circular'] = False
    if(not params['gaussian'] or params['gaussianDiameter'] <= 1):
        params['gaussianDiameter'] = 0
        params['gaussian'] = False
    if(not params['linear'] or params['linearDiameter'] <= 1):
        params['linearDiameter'] = 0
        params['linear'] = False
    if(not params['custom'] or params['customFile'] is None):
        params['customDiameter'] = 0
        params['custom'] = False
    if(params['custom']):
        params['customDiameter'] = max(params['customFile'].shape)
    params['circularDiameter'] = int(params['circularDiameter'])
    params['gaussianDiameter'] = int(params['gaussianDiameter'])
    params['linearDiameter'] = int(params['linearDiameter'])
    
    d = max(params['circularDiameter'] + params['linearDiameter'] - 1,
            params['gaussianDiameter'] + params['linearDiameter'] - 1,
            params['linearDiameter'],
            params['customDiameter'])
    if(params['gaussianDiameter'] > 1):
        d = max(d, 10)
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
    for deconvolveType in ("circular", "gaussian", "linear", "custom"):
        if(deconvolveType == "circular" and params['circular']):
            # Circular
            psf = defocus_kernel(params['circularDiameter'], (params['circularDiameter'] + params['linearDiameter'] - 1)*2) 
            noise = 10**(-0.1*params['circularAmount'])
        elif(deconvolveType == "gaussian" and params['gaussian']):
            # Gaussian
            psf = gaussian_kernel(params['gaussianDiameter'], params['gaussianSpread'], max(10, (params['gaussianDiameter'] + params['linearDiameter'] - 1)*2))
            noise = 10**(-0.1*params['gaussianAmount'])
        elif(deconvolveType == "custom" and params['custom']):
            # Custom
            psf = params['customFile']
            noise = 10**(-0.1*params['customAmount'])
        elif(deconvolveType == "linear" and params['linear'] and not params['circular'] and not params['gaussian']):
            # Linear Only
            psf = motion_kernel(params['linearAngle'], params['linearDiameter'], (params['linearDiameter'] + params['linearDiameter'] - 1)*2)
            noise = 10**(-0.1*params['linearAmount'])
        else:
            continue
        if((deconvolveType == "circular" or deconvolveType == "gaussian") and params['linear']):
            # Stretch&Rotate the Gaussian/Circular
            psf = stretchPSF(psf, (params['linearDiameter']/max(params['gaussianDiameter'],params['circularDiameter'])), params['linearAngle'])
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
