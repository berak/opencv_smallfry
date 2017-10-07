TINY = -DDNN_USE_IMAGE_API=1 -DCNN_USE_OMP=0 -Ic:\p\tiny-dnn
OCV  = -g -Ic:\p\opencv\build -Ic:\p\opencv\build\install\include
LIBOCV = -Lc:\p\opencv\build\install\x64\mingw\staticlib -lopencv_dnn330d -lopencv_ml330 -lopencv_highgui330 -lopencv_imgcodecs330  -lopencv_imgproc330 -lopencv_core330  -llibprotobuf
LIBWIN =  -lkernel32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 -lvfw32 -llibpng -llibjpeg -llibwebp -llibtiff -lopengl32 -lglu32 -luser32 -lz -lgcov
all: tsc transfer
tsc:
	g++ -std=c++11 tsc.cpp $(TINY) $(OCV) $(LIBOCV) $(LIBWIN) -O3 -o tsc
transfer:
	g++ -std=c++11 tsc_transfer.cpp $(OCV) $(LIBOCV) $(LIBWIN) -O3 -o transfer
clean:
	rm tsc transfer
