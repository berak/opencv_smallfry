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
	Mat_<float> D(3,3); D << 1,2,3, 2,3,4, 5,5,5;
	Mat_<float> T(1,3); T << 6,6,6;
	Mat_<int>   L(3,1); L << -1,-1, 1;
	Ptr<ml::Boost> mdl = ml::Boost::create();
	mdl->train(D,0,L);
	float p1 = mdl->predict(T);
	float p2 = mdl->predict(T,noArray(), ml::StatModel::RAW_OUTPUT|ml::Boost::PREDICT_SUM);
	float p3;// = mdl->predictTrees(T,noArray(), ml::StatModel::RAW_OUTPUT);
	cout << p1 << " " << p2 << "  " << p3 << endl;
	return 0;
}
