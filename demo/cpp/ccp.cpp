
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    cv::Mat image = cv::imread("connected.bmp", cv::IMREAD_GRAYSCALE);

	/*Mat_<uchar> m(10,10);
	m << 0,0,0,0,0,0,0,0,0,0,
	     0,0,0,0,0,0,0,0,0,0,
	     0,0,0,0,0,0,1,0,0,0,
	     0,0,0,0,0,0,1,0,0,0,
	     0,1,1,0,0,0,1,0,0,0,
	     0,0,0,0,0,0,0,0,0,0,
	     0,0,0,0,0,0,1,0,0,0,
	     0,0,1,1,0,0,1,1,0,0,
	     0,0,0,0,0,0,0,0,0,0,
	     0,0,0,0,0,0,0,0,0,0;
	Mat image = m;
    */
    cout << image.type() << image.size() << image.depth() << endl;
    cout << image.total() << endl;
    cout << countNonZero(image) << endl;
    cout << (countNonZero(image) > UINT16_MAX) << endl;

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    int numLabels = cv::connectedComponentsWithStats(image, labels, stats, centroids, 8, CV_16U);
    cout << numLabels << endl;
    cout << centroids.size() << endl;
    cout << stats.size() << endl;
	return 0;
 }
