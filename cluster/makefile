INSTALL = c:/p/opencv/build/install
TARGET  = cluster
OBJ     = cluster.o kmeans_flann.o brute_random.o
OBJ_UN  = cluster_un.o whispers.o dbscan.o
CFLAGS  = -std=c++11 -I. -I$(INSTALL)/include
LFLAGS  = -O3 -L$(INSTALL)/x64/mingw/staticlib
OCVLIBS = -lopencv_dnn340 -lopencv_highgui340 -lopencv_imgcodecs340 -lopencv_imgproc340 -lopencv_core340
OCVDEPS = -llibprotobuf -llibtiff -llibpng -llibjpeg -llibwebp -lz
OGLLIBS = -lopengl32 -lglu32 -luser32
WINLIBS = -lkernel32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 -lvfw32
LIBS    = $(OCVLIBS) $(OCVDEPS) $(OGLLIBS) $(WINLIBS)

all: $(TARGET) cluster_un
cluster_un: $(OBJ_UN)
	g++ $(LFLAGS)  $(OBJ_UN) $(LIBS) -o cluster_un
$(TARGET): $(OBJ)
	g++ $(LFLAGS) $(OBJ) $(LIBS) -o $(TARGET)
%.o: %.cpp;
	g++ $(CFLAGS) -c $< -o $@
clean:
	rm -f $(TARGET) $(OBJ) $(OBJ_UN)
