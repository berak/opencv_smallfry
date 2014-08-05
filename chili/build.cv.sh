tar -xf ocv.tgz
g++ -std=c++0x cv.cpp -I ocv/include -L ocv/lib -ljpeg -lpng -ltiff  -ljasper -lopencv_highgui -lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab -lopencv_core -lrt -lz -lpthread -o cv
#cl cv.cpp -I "e:/code/opencv/build/install/include" "E:\code\opencv\build\lib\Release/opencv_core300.lib" "E:\code\opencv\build\lib\Release/opencv_highgui300.lib" 
 
