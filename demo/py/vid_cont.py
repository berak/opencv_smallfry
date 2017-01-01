import numpy as np
import cv2
import copy

cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()
    clone = copy.copy(img)
    gray_image = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image,127,255,1)

    # contours,h = cv2.findContours(thresh,1,2)

    contours, h = cv2.findContours(thresh,1,2)

    for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            if len(approx)==4:
                cv2.drawContours(img,[cnt],0,(0,0,255, 1),-1)
                print len(approx)

    # Display the resulting frame
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
