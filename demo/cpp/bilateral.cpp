#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

int main()
{
	cv::Mat input = cv::Mat::ones(cv::Size(1000, 1000), CV_32F);

	cv::Mat output;// = input;

	cv::bilateralFilter(input, output, 9, 100, 100);
	return 0;
}
