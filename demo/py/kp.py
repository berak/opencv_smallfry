import cv2
import numpy as np

help(cv2.resize)

ocv  = cv2.imread('../im/EaiLU.png')
#ocv  = cv2.resize(ocv, None, fx=2,fy=2)
mser = cv2.MSER_create()
gray = cv2.cvtColor(ocv, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
regions = mser.detectRegions(gray, None)
#print regions

for r in regions:
	if cv2.contourArea(r) > 50:
		continue
	cv2.drawContours(ocv, [r], 0, (200,0,0), 2)

cv2.imshow("I",ocv)
cv2.waitKey()
