#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

int main()
{
	cv::Mat_<float> obj_points(4,3);   obj_points << 0, 0, 0,  0, 1, 0,  1, 1, 0,  1, 0, 0;
	cv::Mat_<float> img_points(4,2);   img_points << 700, 400,  700, 600,  900, 600,  900, 400;
	cv::Mat_<float> cameraMatrix(3,3); cameraMatrix << 712.0634, 0, 800,  0, 712.540, 500,  0, 0, 1;
	cv::Mat_<float> distCoeffs(1,4);   distCoeffs << 0, 0, 0, 0;

	//Mat rep;
	Mat_<float> rep;
	vector<Mat> rvecs,tvecs;
    float r = cv::solvePnPGeneric( obj_points, img_points, cameraMatrix, distCoeffs, rvecs, tvecs, false, SOLVEPNP_UPNP, noArray(), noArray(), rep );
	cout << r << endl << rep << endl;
	return 0;
}

//SOLVEPNP_UPNP C:\p\opencv\modules\calib3d\src\solvepnp.cpp:1017