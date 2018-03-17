import cv2, numpy as np

image = cv2.imread("A.jpg",0)
winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.0
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-0
gammaCorrection = 0
nlevels = 64
hog=cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
winStride = (8,8)
padding = (8,8)
locations = ()
hist = hog.compute(image,winStride,padding,locations)

print(np.sum(hist))
