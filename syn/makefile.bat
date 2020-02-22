all: syn track cam
syn:
	g++ -D__WINDOWS_DS__ -I. syn.cpp  RtAudio.cpp AudioFile.cpp -ldsound -lwinmm -lole32 -o syn

track:
    csc track.cs

cam:
	g++ cam.cpp -I C:/p/opencv/build/install/include -L C:/p/opencv/build/install/x86/mingw/lib -O3 -lopencv_imgcodecs420 -lopencv_videoio420 -lopencv_imgproc420 -lopencv_core420 -lopencv_highgui420 -o cam