import cv2, numpy as numpy

im = cv2.imread("line2.jpg",0)
_,im = cv2.threshold(im,128,255,0)
print(im.shape, im.dtype)
pts = cv2.findNonZero(im)
print(pts.shape)

ends = []
for pt in pts:
    on = 0
    p=pt[0]
    if im[p[1]+1,p[0]-1]>0: on +=1
    if im[p[1]+1,p[0]  ]>0: on +=1
    if im[p[1]+1,p[0]+1]>0: on +=1
    if im[p[1]  ,p[0]-1]>0: on +=1
    if im[p[1]  ,p[0]+1]>0: on +=1
    if im[p[1]-1,p[0]-1]>0: on +=1
    if im[p[1]-1,p[0]  ]>0: on +=1
    if im[p[1]-1,p[0]+1]>0: on +=1
    if on==1: ends.append(p)
    print(p, on)
print (ends)
