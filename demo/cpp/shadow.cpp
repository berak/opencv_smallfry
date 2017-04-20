#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
using namespace cv;
#include <fstream>
using namespace std;

int main(int argc, const char * argv[])
{
	Mat img = imread("shadow2.png");
	Mat ycbcr; cvtColor(img, ycbcr, COLOR_BGR2YCrCb);
	//Mat ycbcr; cvtColor(img, ycbcr, COLOR_BGR2YCrCb);
	Scalar m,d;
	Mat chn[3];
	split(ycbcr, chn);
	Mat Y;
	equalizeHist(chn[0], Y);
	meanStdDev(Y, m, d);
	Mat shadow = Y < d[0];
	Scalar m2,d2;
	meanStdDev(Y, m2, d2, Y > d[0]);
	Scalar m3,d3;
	meanStdDev(Y, m3, d3, Y < d[0]);
	//shadow = Y < d2[0];
	cerr << m << d << endl;
	cerr << m2 << d2 << endl;
	cerr << m3 << d3 << endl;
	imshow("org", img);
	imshow("sha", shadow);
	imshow("mask", Y > d[0]);
	waitKey();
    return 0;
}
