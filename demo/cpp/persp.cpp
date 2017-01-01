#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
    Mat img = imread("oven.jpg");
    vector<Point2f> pts;  // hardcoded, in clockwise order
    pts.push_back(Point2f(167,27));
    pts.push_back(Point2f(367,55));
    pts.push_back(Point2f(382,159));
    pts.push_back(Point2f(182,197));

    Size sz(400,250);    // desired output size
    vector<Point2f> rct; // desired coords, same order
    rct.push_back(Point2f(0,0));
    rct.push_back(Point2f(sz.width,0));
    rct.push_back(Point2f(sz.width,sz.height));
    rct.push_back(Point2f(0,sz.height));

    Mat P = getPerspectiveTransform(pts,rct);
    Mat persp;
    warpPerspective(img, persp, P, sz, INTER_LINEAR);

    imshow("org",img);
    imshow("persp", persp);
    waitKey();
    return 0;
}
