all: tsc
tsc:
	g++ -std=c++11  tsc.cpp -I. -IC:\p\tiny-dnn-master "-DCNN_USE_OMP=0"      -I. -Ic:\p\opencv\build -Ic:\p\opencv\build\install\include -Lc:\p\opencv\build\
install\x64\mingw\staticlib -lopencv_bioinspired310 -lopencv_tracking310 -lopencv_datasets310 -lopencv_face310 -lopencv_xfeatures2d310 -lopencv_features2d310 -lopencv_c
alib3d310 -lopencv_flann310 -lopencv_objdetect310 -lopencv_ml310 -lopencv_photo310 -lopencv_video310 -lopencv_videoio310 -lopencv_highgui310 -lopencv_imgcodecs310 -lope
ncv_ximgproc310 -lopencv_imgproc310 -lopencv_core310 -lkernel32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 -lvfw32 -llibpng -llibjpeg
-llibwebp -lopengl32 -lglu32 -luser32 -lz -lgcov -O3 -o tsc
