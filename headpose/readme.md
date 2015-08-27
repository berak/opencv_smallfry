small head pose estimation using 
    [dlib](http://sourceforge.net/projects/dclib/files/dlib/) (you'll need their shape_predictor_68_face_landmarks.dat, too !)
	and a (slightly opencv adapted version of) [pico](https://github.com/nenadmarkus/pico)
	
<p align="center">
  <img src="https://github.com/berak/opencv_smallfry/raw/master/headpose/pose.png">
</p>

-----------------

the idea is quite simple:

* initially:
  + have a 3d model with face texture
  + run landmark estimation on that face texture, so we can
    take the corresponding 3d points from the model as our base.
	
* then, for each frame:
  + face detection, if successful,
  + landmark estimation on cameraframe
  + solvePnP with the landmarks, and the base 3d points, to get rvec and tvec.
