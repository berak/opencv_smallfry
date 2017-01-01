#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;

int main( int argc, char** argv )
{

    namedWindow("mask", 0);
    CV_Assert(argc>7);
    String c_in = "."; // check a whole folder.
    if (argc>1) c_in = argv[1];
    double v[6];
    for (int i=2; i<argc; i++)
    {
        v[i-2] = atof(argv[i]);
    }
    // v[3] *= v[3];
    // v[4] *= v[4];
    // v[5] *= v[5];
    double scale = 1.5;
    v[3] *= scale;
    v[4] *= scale;
    v[5] *= scale;
    Scalar lo(v[0]-v[3], v[1]-v[4], v[2]-v[5]);
    Scalar hi(v[0]+v[3], v[1]+v[4], v[2]+v[5]);
    cerr << lo << hi << endl;

    cerr << c_in << endl;
    Mat im_bgr = imread(c_in);
    if (im_bgr.empty()) return -1;
    Mat im_hsv;
    cvtColor(im_bgr, im_hsv, COLOR_BGR2HSV);
    blur(im_hsv, im_hsv, Size(8,8));
    Mat mask;
    inRange(im_hsv, lo, hi, mask);
    int nz = mask.total() - countNonZero(mask);
    float cov = float(nz) / float(mask.total());
    cerr << nz << " pixels, " << cov << " ratio." << endl;
    imshow("orig", im_bgr);
    im_bgr.setTo(Scalar(0,255,255), mask);
    imshow("mask", im_bgr);
    waitKey();
    return 0;
}
