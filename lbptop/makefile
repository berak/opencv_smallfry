TINY = -DDNN_USE_IMAGE_API=1 -DCNN_USE_OMP=0 -Ic:\p\tiny-dnn
INSTALL = c:/p/opencv/build/install
CFLAGS  = -std=c++11 -I. -I$(INSTALL)/include
LFLAGS  = -O3 -L$(INSTALL)/x64/mingw/staticlib
OCVLIBS = -lopencv_videoio331 -lopencv_ml331 -lopencv_objdetect331 -lopencv_highgui331 -lopencv_imgcodecs331 -lopencv_imgproc331 -lopencv_core331
OCVDEPS = -llibtiff -llibpng -llibjpeg -llibwebp -lz
OGLLIBS = -lopengl32 -lglu32
WINLIBS = -luser32 -lkernel32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 -lvfw32
LIBS    = $(OCVLIBS) $(OCVDEPS) $(OGLLIBS) $(WINLIBS)


all: cam hockey aslan ckplus ckplus_lm ckplus_tiny
cam: cam.o lbptop.o
	g++ $(LFLAGS) cam.o lbptop.o $(LIBS) -o cam
hockey: hockey.o lbptop.o
	g++ $(LFLAGS) hockey.o lbptop.o $(LIBS) -o hockey
aslan: aslan.o lbptop.o
	g++ $(LFLAGS) aslan.o lbptop.o $(LIBS) -o aslan
ckplus: ckplus.o lbptop.o
	g++ $(LFLAGS) ckplus.o lbptop.o $(LIBS) -o ckplus
ckplus_tiny:
	g++ $(TINY) $(CFLAGS) ckplus_tiny.cpp $(LFLAGS) -lopencv_tracking331 $(LIBS) -o ckplus_tiny
ckplus_lm: ckplus_lm.o
	g++ $(CFLAGS) $(LFLAGS) ckplus_lm.o dliblm.cpp c:/p/dlib-19.0/dlib/all/source.cpp -Ic:\p\dlib-19.0  -lws2_32 -lwinmm -lcomctl32 -limm32 $(LIBS) -o ckplus_lm
%.o: %.cpp;
	g++ $(CFLAGS) -c $< -o $@
clean:
	rm -f *.exe *.o *.png *.net
