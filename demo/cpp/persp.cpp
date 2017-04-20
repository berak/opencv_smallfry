#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
    Mat img = imread("maxresdefault.jpg");
    vector<Point2f> pts;  // hardcoded, in clockwise order
    pts.push_back(Point2f(895,204));
    pts.push_back(Point2f(995,195));
    pts.push_back(Point2f(1222,292));
    pts.push_back(Point2f(1099,307));

    Size sz(250,500);    // desired output size
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
