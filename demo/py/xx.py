import numpy as np
import cv2
help(cv2.drawKeypoints)
img = cv2.imread('plank.jpg',0)
print np.shape(img)
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img,None)
print len(kp)
img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
cv2.imshow("W",img2)
cv2.waitKey()
