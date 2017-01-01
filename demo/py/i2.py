import cv2, numpy as np

im = cv2.imread("key1.jpg")

roi1 = im[20:40, 20:40]
im[60:80,60:80] = roi1
cv2.imshow("W",im)
cv2.waitKey()
