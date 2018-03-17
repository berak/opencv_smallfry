import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

while(vc.isOpened()): # try to get the first frame
    rval, frame = vc.read()
    if not rval:
    	break
    cv2.imshow("preview", frame)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")
