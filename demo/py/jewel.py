import cv2
import numpy as np

im=cv2.imread('jewel.jpg')
#im = cv2.resize(im,(500,500))
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#res = cv2.GaussianBlur(thresh, ksize = ( 3,3 ), sigmaX = 0 )
res = cv2.blur(thresh,(6,6))
#res = res > 128
ret, res = cv2.threshold(res,15,255,0)

kernel = np.ones((3,3),dtype='uint8')
#res = cv2.dilate(res, kernel, iterations = 1)
res = cv2.erode(res,kernel,iterations = 1)
res = cv2.dilate(res,kernel,iterations = 1)
#res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
#res = thresh

cv2.imshow('thresh',res)

_,contours, hierarchy = cv2.findContours(res.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


(wi,hi) = np.shape(res)
print wi*hi
for i in range(len(contours)):
    a = cv2.contourArea(contours[i])
    print a
    if a >= wi*hi*0.4:
        print "exclude ", a
        continue
    x,y,w,h = cv2.boundingRect(contours[i])
    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
