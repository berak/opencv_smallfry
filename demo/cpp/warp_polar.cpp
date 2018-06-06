#include <opencv2/opencv.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/core/cv_cpu_helper.h>

#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <iostream>
//#include "profile.h"

using namespace std;
using namespace cv;
#include <chrono>
#include <thread>

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
	Mat ocv = imread("360fly.jpg");
	resize(ocv,ocv,Size(),0.5,0.5);
	imshow("org",ocv);
	Size dsize(ocv.cols, ocv.rows*2);

	Point2f center(ocv.cols/2, ocv.rows/2);

	float maxRadius = ocv.rows/2;

	int flags = WARP_POLAR_LINEAR;

	Mat dst;
	warpPolar(ocv,dst,dsize,center,maxRadius,flags);
	rotate(dst,dst,ROTATE_90_CLOCKWISE);

	imshow("warp",dst);
	waitKey();
	return 0;
}
