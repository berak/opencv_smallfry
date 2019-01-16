cmake_minimum_required(VERSION 2.8)

find_package(OpenCV  REQUIRED)

add_executable( benchmark benchmark.cpp ./cmt/CMT.cpp ./bacf/bacf.cpp ./cmt/fastcluster/fastcluster.cpp ./staple/fhog.cpp ./staple/staple_tracker.cpp ./ct/CompressiveTracker.cpp)

if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	target_link_libraries( benchmark ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)