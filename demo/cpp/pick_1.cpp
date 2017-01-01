#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;

Mat im_hsv;
void pick_color(int e, int x, int y, int s, void *)
{
    if (e==1) // left mouse down
    {
        Vec3b p = im_hsv.at<Vec3b>(y, x);
        cerr << int(p[0]) << " " << int(p[1]) << " " << int(p[2]) << endl;
    }
}

// use: pick.exe my.png
int main( int argc, char** argv )
{
    namedWindow("hsv");
    setMouseCallback("hsv", pick_color);

    if (argc<2) return -1;
    Mat im_bgr = imread(argv[1]);
    if (im_bgr.empty()) return -2;

    cvtColor(im_bgr, im_hsv, COLOR_BGR2HSV);
    imshow("hsv", im_hsv);
    waitKey();

    return 0;
}
