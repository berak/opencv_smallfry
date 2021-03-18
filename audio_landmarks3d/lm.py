import numpy as np
import cv2
import wave
import subprocess
import os, base64

# landmarks connections
cons = [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57],
        [57, 58], [58, 59], [59, 48], [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66],
        [66, 67], [67, 60], [27, 28], [28, 29], [29, 30], [30, 31], [30, 35], [31, 32], [32, 33],
        [33, 34], [34, 35], [27, 31], [27, 35], [17, 18], [18, 19], [19, 20], [20, 21],
        [22, 23], [23, 24], [24, 25], [25, 26], [36, 37], [37, 38], [38, 39], [39, 40], [40, 41],
        [36, 41], [43, 44], [44, 45], [45, 46], [46, 47], [42, 47], [0, 1], [1, 2], [2, 3], [3, 4],
        [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
        [12, 13], [13, 14], [14, 15], [15, 16]]

sr = 8000 # assumes: 8khz mono
num_frames = 7
increment  = sr * 0.04 # 25 fps
W,H = 400,400; # drawing

net = cv2.dnn.readNet("model.onnx")
mean_shape = np.load("mean_shape.npy")
eigen_vectors = np.load("eigen_vectors.npy").T

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
		pts = pts.reshape(68,3) # 204==68*3
		img = np.ones((H,W,3),np.uint8)
		img[:,:] = 127
		for i in range(pts.shape[0]):
			x = int(pts[i,0] * W*2 + W/2)
			y = int(pts[i,1] * H*2 + H/2)
			cv2.circle(img, (x,y), 3, (50,50,255), -1)
		for c in cons:
			x1 = int(pts[c[0],0] * W*2 + W/2)
			y1 = int(pts[c[0],1] * H*2 + H/2)
			x2 = int(pts[c[1],0] * W*2 + W/2)
			y2 = int(pts[c[1],1] * H*2 + H/2)
			cv2.line(img,(x1,y1),(x2,y2),(20,20,180),1)
		vid.write(img)
		cv2.imshow("draw", img)
		cv2.waitKey(6)

	vid.release()
	#cv2.waitKey();

	cmd = 'ffmpeg -y -i my.avi -i '+wfile+' -c:v copy -c:a aac -strict experimental res_.webm'
	subprocess.call(cmd, shell=True)

animate("S2.wav")
"""
def application(environ, start_response):
    request_body=None
    retcode = '200 OK'
    resp = "dummy\r\n"
    ct  ="text/html"
    try:
       request_body_size = int(environ.get('CONTENT_LENGTH', 0))
       request_body = environ['wsgi.input'].read(request_body_size)
    except (ValueError):
       resp = "no response"
    url = environ['PATH_INFO'];
    if url == "/":
        resp = _read("up.html")
    elif url == "/dn":
        ct = 'image/png'
        resp = _read("my.png")
    elif url == "/up" and request_body:
        ct = 'image/png'
        resp = request_body.replace('data:' + ct + ';base64,', "")
        data = base64.b64decode(resp)
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, 1)
        img = process(img)
        cv2.imwrite("my.png", img)
        ok, enc = cv2.imencode(".png", img)
        resp = base64.b64encode(enc.tostring())
        resp = 'data:' + ct + ';base64,' + resp
    start_response(retcode, [('Content-Type', ct), ('Content-Length', str(len(resp)))])
    return [resp]

if __name__ == '__main__':
    from wsgiref.simple_server import make_server
    httpd = make_server('0.0.0.0', int(os.environ.get("PORT", 9000)), application)
    while True: httpd.handle_request()
"""