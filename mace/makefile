INSTALL = c:/p/opencv/build/install
CFLAGS  = -std=c++11 -I. -Ic:/p/opencv/build -I$(INSTALL)/include
LFLAGS  = -O3 -L$(INSTALL)/x64/mingw/staticlib
OCVLIBS = -lopencv_highgui340 -lopencv_imgcodecs340 -lopencv_imgproc340 -lopencv_core340
OCVDEPS = -llibtiff -llibpng -llibjpeg -llibwebp -lz
OGLLIBS = -lopengl32 -lglu32
WINLIBS = -luser32 -lkernel32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 -lvfw32
LIBS    = $(OCVLIBS) $(OCVDEPS) $(OGLLIBS) $(WINLIBS)


all: att webcam fingerprint test
test: test_main.o mace.o
	g++ $(LFLAGS) mace.o test_main.o  -lopencv_videoio340  -lopencv_objdetect340 -Lc:\p\opencv\build\lib -lopencv_ts340  $(LIBS) -o test
att: att.o mace.o
	g++ $(LFLAGS) mace.o att.o $(LIBS) -o att
webcam: webcam.o mace.o
	g++ $(LFLAGS) mace.o webcam.o -lopencv_videoio340 -lopencv_objdetect340 $(LIBS) -o webcam
fingerprint: fingerprint.o mace.o
	g++ $(LFLAGS) mace.o fingerprint.o $(LIBS) -o fingerprint
%.o: %.cpp;
	g++ $(CFLAGS) -c $< -o $@
clean:
	rm -f *.exe *.o
