import base64
import wave
import subprocess
import numpy as np
import sys
print("path pre",sys.path)
import cv2
print(cv2.__version__)

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
    img[:,:] = (127,127,127)
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

  vid.release()

  cmd = 'ffmpeg -y -i my.avi -i '+wfile+' -c:v h264 -c:a aac -strict experimental res_.mp4 > /dev/null'
  subprocess.call(cmd, shell=True)



HELLO_WORLD = b"""
<!Doctype html>
<html>
<head>
    <title>facial landmarks from audio</title>
</head>
<body>
<div id="droparea">
  <p> drop an audio sample (8khz, 16bit mono, wav) here or load from disk</p>
  <p> <input type="file" accept="audio/*" capture id="recorder"></p>
  <p> <div id="err"></div></p>
</div>


<script type="text/javascript">

  var recorder = document.getElementById("recorder")
  recorder.addEventListener('change', function(e) {
    const file = e.target.files[0];
    postSoundToURL("/up", file);
  });

  function postSoundToURL(url, data) { // this is the actual workhorse
    err.innerHTML = "... posting sound";
    var type = "audio/wav"
    var xhr = new XMLHttpRequest();
    xhr.open('POST', url, true);
    xhr.setRequestHeader('Content-Type', type);
    xhr.onreadystatechange = function(e) {
      err.innerHTML = "... " + this.readyState + " " + e
      if ( this.readyState > 3 ) {
        err.innerHTML = this.responseText;
      }
    }

    var reader = new FileReader();
    reader.onload = function(e){
      res = e.target.result.replace('data:' + type + ';base64,', '');
      xhr.send(res);
    };
    reader.readAsDataURL(data);
  }

  var target = document.getElementById("droparea");
  target.addEventListener("dragover", function(e){e.preventDefault();}, true);
  target.addEventListener("drop", function(e){
    e.preventDefault();
    postSoundToURL("/up", e.dataTransfer.files[0]);
  }, true);


</script>
</body>
</html>
"""

VIDEO = b"""
<html>
<video id="vid" width="400" height="400" controls>
  <source src="res_.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>
</html>
"""

def _read(fname):
    try:
        f = open(fname,"rb")
        r = f.read()
        f.close()
        return r
    except:
        return ""

def application(environ, start_response):
    request_body_size=0
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
    print(url,request_body_size)
    if url == "/":
        resp = HELLO_WORLD
    elif url == "/res_.mp4":
        ct = 'video/mp4'
        resp = _read("res_.mp4")
    elif url == "/vid":
        resp = VIDEO
    elif url == "/up" and request_body_size>0:
        resp = VIDEO
        res  = request_body #.replace(b'data:' + ct + b';base64,', b"")
        data = base64.b64decode(res)
        f = open("S.wav","wb")
        f.write(data)
        f.close()
        animate("S.wav")
    else:
        resp = "404 - file "+url+" not found"
        retcode = "404 ERROR"
    headers = [('Content-type', ct), ('Content-Length', str(len(resp)))]
    start_response(retcode, headers)
    return [resp]

