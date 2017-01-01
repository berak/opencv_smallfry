#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
    Mat img = imread("qsutdl.png");
    Mat imgGray;
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    Mat thresholded;

    GaussianBlur(imgGray, imgGray, Size(5,5), 0, 0);
    adaptiveThreshold(imgGray, thresholded, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 4);
    imshow("img", thresholded); waitKey(0);
    bitwise_not(thresholded,thresholded);
    imshow("img", thresholded); waitKey(0);

    vector<Vec4i> lines;
    HoughLinesP(thresholded, lines, 1, CV_PI/180, 70, 100, 10);

    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        line( img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,255,0), 3, LINE_AA);

    }

    imshow("houghlines", img); waitKey(0);

    return 0;
}
