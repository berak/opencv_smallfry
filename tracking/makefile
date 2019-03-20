all:
	g++ -fno-diagnostics-color  benchmark.cpp \
		 ct/CompressiveTracker.cpp \
		 cmt/CMT.cpp cmt/fastcluster/fastcluster.cpp \
		 bacf/bacf.cpp \
		 staple\staple_tracker.cpp staple\fhog.cpp \
		 dat/dat_tracker.cpp \
		 -I. -Ic:\p\opencv\build\install\include -Lc:\p\opencv\build\install\x64\mingw\lib \
		 -lopencv_tracking410 -lopencv_plot410 -lopencv_datasets410 -lopencv_dnn410 -lopencv_features2d410 -lopencv_flann410 -lopencv_video410 -lopencv_videoio410 -lopencv_highgui410 -lopencv_imgcodecs410 -lopencv_imgproc410 -lopencv_core410 -DNDEBUG -O3 -o benchmark
