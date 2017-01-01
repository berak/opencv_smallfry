import cv2, numpy as np

li = [np.zeros((3,3),np.uint8),np.ones((3,3),np.uint8),np.ones((3,3),np.uint8)*3]

print(li)

li = np.asarray(li, np.float32)
N = len(li)
li = li.reshape(N,-1)

print(li)

K = 2
term_crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
ret,labels,centers = cv2.kmeans(li, K, None, term_crit, 3, 0)
#cv2.kmeans(li, 2, None, term_crit, 10, 0)
print (centers)
