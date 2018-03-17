import numpy as np
import cv2 as cv
img = cv.imread('c:/p/opencv/samples/data/fruits.jpg',cv.IMREAD_COLOR)
orange = img[300:380,150:220]
cv.imshow('fruits',img)
cv.imshow('orange',orange)
# calculating object histogram
if 0:
  c=[0,1,2]
  r=[32,32,32]
  rc=[0,256,0,256,0,256]
else:
  c=[1,2]
  r=[32,32]
  rc=[0,256,0,256]
roihist = cv.calcHist([orange],c, None, r, rc )
print(np.sum(roihist[0]),np.sum(roihist[1]),np.sum(roihist[2]))

# normalize histogram and apply backprojection
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
dst = cv.calcBackProject([img],c,roihist,rc,1)
ret,thresh = cv.threshold(dst,10,255,0)
thresh = cv.merge((thresh,thresh,thresh))
cv.imshow('dst',dst)
cv.imshow('res',thresh)
cv.waitKey(0)
cv.destroyAllWindows()
