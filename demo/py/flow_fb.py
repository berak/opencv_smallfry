import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while (1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 2, 5, 1.2, 0)
    print flow.shape
    cv2.imshow("W",frame2)
    k = 0xff & cv2.waitKey(40)
    if k == 27: break
    print "Flow"
    print flow
    print "Flow : : 0"
    print flow[:,:,0]
    print "Flow : : 1"
    print flow[:,:,1]
    break
