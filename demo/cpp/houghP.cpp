
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    int sz = 512;
    Mat img(sz, sz, CV_8U, Scalar(0));
    Point point1(0, 0);
    Point point2(sz, sz);
    Point point3(sz, 0);
    Point point4(2*sz/3, sz/3);
    line(img, point1, point2, Scalar(255), 1);
    line(img, point3, point4, Scalar(255), 1);

    vector<Vec4i> lines;
    HoughLinesP(img, lines, 1, 3.1415926/180, 100);
    cerr << Mat(lines) << endl;
	return 0;
}
