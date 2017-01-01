#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;


int main()
{
    Mat img = imread("small1.png", 0); // load gray
    Rect region(10,10,40,40); // example roi
    Mat roi(img, region);  // part of img
    Mat mask(Size(40,40), CV_8U, Scalar(0)); // all black
    circle(mask, Point(20,20), 20, Scalar(255), -1, LINE_AA); // filled circle
    Mat circRoi;
    bitwise_and(roi, roi, circRoi, mask);
    imshow("circle masked", mask);
    imshow("masked roi", circRoi);
    waitKey();
    return 0; //test_eq();
}
