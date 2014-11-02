#tar -xf ocv.tgz
g++ -std=c++0x cv.cpp -I ocv/include -L ocv/lib -lopencv_highgui -lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab -lopencv_legacy -lopencv_imgproc -lopencv_core -ljpeg -lpng -ltiff -lrt -lz -lpthread -o cv
#cl cv.cpp -I "e:/code/opencv242/include" "E:\code\opencv242\lib\Release/opencv_core242.lib" "E:\code\opencv242\lib\Release/opencv_highgui242.lib" 
 
