import numpy as np
import cv2
import wave
import subprocess

sr = 8000 # assumes: 8khz mono
num_frames = 7
increment  = sr * 0.04 # 25 fps
W,H=400,400; # drawing

net = cv2.dnn.readNet("model.onnx")
mean_shape = np.load("mean_shape.npy")
eigen_vectors = np.load("eigen_vectors.npy").T
print(mean_shape.shape, eigen_vectors.shape)

def animate(wfile):
	w = wave.open(wfile,"rb")
	n = w.getnframes()
	b = w.readframes(n)
	a = np.frombuffer(b,np.int16)
	a = np.array(a,np.float32)
	a /= 0x7ffff
	a /= a.max()

	sample_len = int(num_frames * increment)
	sample_pos = int(0)


	vid = cv2.VideoWriter("my.avi",cv2.VideoWriter_fourcc(*'MJPG'), 25.0, (W,H))
	while (sample_pos < n - sample_len):
		data = a[int(sample_pos):int(sample_pos+sample_len)].reshape(1,1,sample_len)
		sample_pos += increment;
		net.setInput(data)
		res = net.forward()
		pts = mean_shape.copy()
		for i in range(eigen_vectors.shape[0]):
			pts[0,i] += res.dot(eigen_vectors[i,:])
		pts = pts.reshape(68,3)
		img = np.ones((H,W,3),np.uint8)
		print(img.shape)
		img[:,:] = 127
		for i in range(pts.shape[0]):
			x = int(pts[i,0] * W*2 + W/2)
			y = int(pts[i,1] * H*2 + H/2)
			cv2.circle(img, (x,y), 3, (50,50,255), -1)
		vid.write(img)
		cv2.imshow("draw", img)
		cv2.waitKey(60)

	vid.release()
	cv2.waitKey();

	cmd = 'ffmpeg -i my.avi -i '+wfile+' -c:v copy -c:a aac -strict experimental res_.mp4'
	subprocess.call(cmd, shell=True)

animate("S2.wav")