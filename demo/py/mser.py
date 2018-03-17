import cv2
import numpy as np

ocv  = cv2.imread('digi.jpg')
#ocv  = cv2.resize(ocv, None, fx=2,fy=2)
mser = cv2.MSER_create(3,60,1000)
gray = ocv#cv2.cvtColor(ocv, cv2.COLOR_BGR2GRAY)
#gray = cv2.equalizeHist(gray)
contours, regions = mser.detectRegions(gray)
#print regions
print(regions)
for r in regions:
	cv2.rectangle(ocv, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), (200,0,0), 2)
	print(r)
cv2.imshow("I",ocv)
cv2.waitKey()
