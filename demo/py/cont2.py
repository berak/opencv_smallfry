import cv2
import numpy as np

camera = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
while(True):
    # Capture frame-by-frame
    ret, frame = camera.read()
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    fgmask = fgbg.apply(blurred)


    #ret,thresh = cv2.threshold(img,127,255,0)
    im2,contours,hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if (len(contours) != 0):
        maxlen=0
        maxid=0
        for i in range(len(contours)):
            a = cv2.contourArea(contours[i])
            if a > maxlen:
                maxlen = a
                maxid = i
        cnt = contours[maxid]
        print(maxid)
        M = cv2.moments(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        print(x,y,w,h)
        '''if (M['m00'] != 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print str(cx)+" "+str(cy)
            cv2.line(frame, (cx, cy), (cx, cy), (255, 0, 0 ), thickness=2, lineType=8, shift=0)'''
    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('no back',fgmask)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything done, release the capture
#camera.release()
#v2.destroyAllWindows()
