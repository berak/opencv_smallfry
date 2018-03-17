#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int _ddepth = CV_32F, _dx=1,_dy=1,_ksize=3,_scale=1, _delta=0, _borderType=BORDER_DEFAULT;

Mat ridge1(const Mat &img) {

    Mat sbx, sby;
    Sobel(img, sbx, _ddepth, _dx, 0, _ksize, _scale, _delta, _borderType);
    Sobel(img, sby, _ddepth, 0, _dy, _ksize, _scale, _delta, _borderType);

    Mat sbxx, sbyy, sbxy;
    Sobel(sbx, sbxx, _ddepth, _dx, 0, _ksize, _scale, _delta, _borderType);
    Sobel(sby, sbyy, _ddepth, 0, _dy, _ksize, _scale, _delta, _borderType);
    Sobel(sbx, sbxy, _ddepth, 0, _dy, _ksize, _scale, _delta, _borderType);

    Mat dxxyy = sbxx - sbyy;
    Mat d2xxyy, sb2xy;
    multiply(dxxyy, dxxyy, d2xxyy);
    multiply(sbxy, sbxy, sb2xy);

    Mat rootex = d2xxyy + 4*sb2xy;
    Mat root; sqrt(rootex,root);
    Mat res = root - dxxyy;
    res.convertTo(res,CV_8U,0.5);
	return res;
}

Mat ridge2(const Mat &img) {

    Mat sbx, sby;
    Sobel(img, sbx, _ddepth, _dx, 0, _ksize, _scale, _delta, _borderType);
    Sobel(img, sby, _ddepth, 0, _dy, _ksize, _scale, _delta, _borderType);

    Mat sbxx, sbyy, sbxy;
    Sobel(sbx, sbxx, _ddepth, _dx, 0, _ksize, _scale, _delta, _borderType);
    Sobel(sby, sbyy, _ddepth, 0, _dy, _ksize, _scale, _delta, _borderType);
    Sobel(sbx, sbxy, _ddepth, 0, _dy, _ksize, _scale, _delta, _borderType);


    Mat sb2xx, sb2yy, sb2xy;
    multiply(sbxx, sbxx, sb2xx);
    multiply(sbyy, sbyy, sb2yy);
    multiply(sbxy, sbxy, sb2xy);

    Mat sbxxyy;
    multiply(sbxx, sbyy, sbxxyy);
    Mat rootex;
    rootex = (sb2xx +  (sb2xy + sb2xy + sb2xy + sb2xy)  - (sbxxyy + sbxxyy) + sb2yy );
    Mat root;
    sqrt(rootex, root);
    Mat ridgexp, out;
    ridgexp = ( (sbxx + sbyy) + root );
    ridgexp.convertTo(out, CV_8U, 0.5);
	return out;
}

int main(int argc, char** argv)
{

	Mat img = imread("04.bmp", 0);
    imshow("S1",ridge1(img));
    imshow("S2",ridge2(img));
    waitKey();
    return 0;
}
