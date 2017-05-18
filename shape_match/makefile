CF = -std=c++11 -I. -Ic:/p/opencv/build -Ic:/p/opencv/build/install/include
LF = -O3 -Lc:/p/opencv/build/install/x64/mingw/staticlib
LIBS = -lopencv_calib3d320 -lopencv_highgui320 -lopencv_imgcodecs320 -lopencv_imgproc320 -lopencv_core320 -lkernel32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 -lvfw32 -llibtiff -llibpng -llibjpeg -llibwebp -lopengl32 -lglu32 -luser32 -lz
OBJ = match.o hausdorff.o fourier.o onedollar.o sampson.o


all: match
match: $(OBJ)
	g++ $(LF) $(OBJ) $(LIBS) -o match
%.o: %.cpp;
	g++ $(CF) -c  $< -o $@
