import numpy as np
import cv2

cap = cv2.VideoCapture(0)

ret, last_frame = cap.read()

if last_frame is None:
    exit()

while(cap.isOpened()):
    ret, frame = cap.read()

    if frame is None:
        exit()
    vRow, vCol, vCH = frame.shape

#    binary = np.zeros((vRow,vCol), np.uint8)
    binary = cv2.absdiff(last_frame, frame)
    binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(binary,50,255,cv2.THRESH_BINARY)
    #Rectangle
    last_frame = frame.copy()
    cv2.rectangle(frame,(0,0),(320,240),(0,255,0),3)

    #Circle
    cirR = 30
    cirRowPos = int(vRow/2)
    cirColPos = int(vCol-cirR) - int(vCol*0.05)
    cv2.circle(frame,(cirColPos, cirRowPos), cirR, (255,255,0), 3)

    cv2.imshow('binary', binary)
    cv2.imshow('frame', frame)

    if cv2.waitKey(33) >= 0:
        break


cap.release()
cv2.destroyAllWindows()
