INSTALL = c:/p/opencv/build/install
CFLAGS  = -std=c++11 -I. -I$(INSTALL)/include
LFLAGS  = -O3 -L$(INSTALL)/x64/mingw/staticlib
OCVLIBS = -lopencv_videoio331 -lopencv_ml331 -lopencv_objdetect331 -lopencv_highgui331 -lopencv_imgcodecs331 -lopencv_imgproc331 -lopencv_core331
OCVDEPS = -llibtiff -llibpng -llibjpeg -llibwebp -lz
OGLLIBS = -lopengl32 -lglu32
WINLIBS = -luser32 -lkernel32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 -lvfw32
LIBS    = $(OCVLIBS) $(OCVDEPS) $(OGLLIBS) $(WINLIBS)


all: cam hockey aslan ckplus
cam: cam.o lbptop.o
	g++ $(LFLAGS) cam.o lbptop.o $(LIBS) -o cam
hockey: hockey.o lbptop.o
	g++ $(LFLAGS) hockey.o lbptop.o $(LIBS) -o hockey
aslan: aslan.o lbptop.o
	g++ $(LFLAGS) aslan.o lbptop.o $(LIBS) -o aslan
ckplus: ckplus.o lbptop.o
	g++ $(LFLAGS) ckplus.o lbptop.o $(LIBS) -o ckplus
%.o: %.cpp;
	g++ $(CFLAGS) -c $< -o $@
clean:
	rm -f *.exe *.o
