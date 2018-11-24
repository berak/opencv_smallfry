import tensorflow as tf, cv2, numpy as np
ocv = cv2.imread('input.img',-1)
 #cv2.

print ("elo")
if np.shape(ocv)!=():
  cv2.imwrite('output.png',ocv)
