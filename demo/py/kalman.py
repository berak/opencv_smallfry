import cv2
import numpy as np

kalman = cv2.KalmanFilter(4,2)
measurement = np.array((2,1), np.float32)
pts_mouse = []
pts_track = []

def onmouse(b,x,y,s,nada):
    global pts_mouse, pts_track, measurement
    if s==1: # lbuttondown
        pts_mouse.append((x,y))
        measurement[0]=x
        measurement[1]=y

    if b==2: # rbuttonpress
        pts_mouse = []
        pts_track = []

def main():
    cv2.namedWindow("mouse",0)
    cv2.setMouseCallback("mouse", onmouse)

    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.003
    kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 10
    kalman.errorCovPost = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.1

    while(True):
        prediction = kalman.predict()
        estimated = kalman.correct(measurement);

        #pts_track.append((prediction[0],prediction[1]))
        pts_track.append((estimated[0],estimated[1]))

        draw = np.zeros((400,400,3),np.uint8)
        if len(pts_mouse)>0:
            lp = pts_mouse[0]
            for p in pts_mouse:
                cv2.line(draw, lp, p, (200,0,0), 1)
                lp = p
            lp = pts_track[0]
            for p in pts_track:
                cv2.line(draw, lp, p, (0,0,200), 1)
                lp = p

        cv2.imshow("mouse", draw)
        if ((0xff & cv2.waitKey(1)) == 27):
            break
    cv2.destroyAllWindows()

main()
