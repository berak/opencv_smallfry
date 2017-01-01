import numpy as np
import cv2

im = cv2.imread('py/shap.png')
cv2.imshow('original image',im)
cv2.waitKey()

imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray image',imgray)
cv2.waitKey()

ret,thresh = cv2.threshold(imgray,80,255,1)
cv2.imshow('thresh image',thresh)
cv2.waitKey()

image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('cont image',image)
cv2.waitKey()
ctt=0
for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    print len(approx)
    if len(approx)==5:
        print "pentagon"
        cv2.drawContours(im,[cnt],0,255,-1)
        ctt+=1
    elif len(approx)==3:
        print "triangle"
        cv2.drawContours(im,[cnt],0,(0,255,0),-1)
        ctt+=1
    elif len(approx)==4:
        print "square"
        cv2.drawContours(im,[cnt],0,(0,0,255),-1)
        ctt+=1
    elif len(approx) == 9:
        print "half-circle"
        cv2.drawContours(im,[cnt],0,(255,255,0),-1)
        ctt+=1
    elif len(approx) > 15:
        print "circle"
        cv2.drawContours(im,[cnt],0,(0,255,255),-1)
        ctt+=1
    else:
        print "unknown"

cv2.imshow('img',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
print "Total no.= ",ctt
