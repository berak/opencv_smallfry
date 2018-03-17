import cv2
import numpy as np

cv2.namedWindow("bin",0)
img = cv2.imread("kCIAMyI.jpg");
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
bin = cv2.inRange(hsv,(56,189,137),(76,209,217))
bin = cv2.dilate(bin,())
bin = cv2.dilate(bin,())

cv2.imshow("bin",bin);
cv2.waitKey();

_,contours, hierarchy = cv2.findContours(bin,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

print(len(contours))
pts = []
if len(contours) == 4:
	for c in contours:
		M = cv2.moments(c)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		pts.append((cx,cy))

print(pts)
