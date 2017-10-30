INSTALL = c:/p/opencv/build/install
TARGET  = match
OBJ     = match.o hausdorff.o fourier.o onedollar.o sampson.o
CFLAGS  = -std=c++11 -I. -I$(INSTALL)/include
LFLAGS  = -O3 -L$(INSTALL)/x64/mingw/staticlib
OCVLIBS = -lopencv_calib3d331 -lopencv_highgui331 -lopencv_imgcodecs331 -lopencv_imgproc331 -lopencv_core331
OCVDEPS = -llibtiff -llibpng -llibjpeg -llibwebp -lz
OGLLIBS = -lopengl32 -lglu32 -luser32
WINLIBS = -lkernel32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 -lvfw32
LIBS    = $(OCVLIBS) $(OCVDEPS) $(OGLLIBS) $(WINLIBS)


all: $(TARGET)
$(TARGET): $(OBJ)
	g++ $(LFLAGS) $(OBJ) $(LIBS) -o $(TARGET)
%.o: %.cpp;
	g++ $(CFLAGS) -c $< -o $@
clean:
	rm -f $(TARGET) $(OBJ)
