#include <set>
#include <iostream>
#include <opencv2/opencv.hpp>
//#include "xui.h"

using namespace cv;
using namespace std;


int main(int  argc, char **argv)
{
    int h=30, w = 100, wid=10, sig=12, tet=w/2, lam=w/2, gam=w/2, ps=w/2;
    double scal = 1.0 / double(w);

  /*  Ptr<UI> ui = createUI("Gabor");
    ui->addSlider("width", Rect(30,h*1,w,30), Scalar(90,0,120), wid);
    ui->addSlider("sigma", Rect(30,h*2,w,30), Scalar(90,0,120), sig);
    ui->addSlider("theta", Rect(30,h*3,w,30), Scalar(90,0,120), tet);
    ui->addSlider("lambda",Rect(30,h*4,w,30), Scalar(90,0,140), lam);
    ui->addSlider("gamma", Rect(30,h*5,w,30), Scalar(90,0,140), gam);
    ui->addSlider("psi",   Rect(30,h*6,w,30), Scalar(120,20,60), ps);
*/
    namedWindow("sliders", 0);
    createTrackbar("width","sliders",&wid,w);
    createTrackbar("sigma","sliders",&sig,w);
    createTrackbar("theta","sliders",&tet,w);
    createTrackbar("lambda","sliders",&lam,w);
    createTrackbar("gamma","sliders",&gam,w);
    createTrackbar("psi","sliders",&ps,w);

    String fn="key1.jpg";
    if (argc > 1) fn = argv[1];
    Mat frame = imread(fn, 0);
    resize(frame,frame,Size(640,480));

    while(1)
    {
    	double sigma = double(sig) * scal * 12;
    	double theta = double(tet) * scal / (2*CV_PI);
    	double lambda = double(lam) * scal;
    	double gamma = double(gam) * scal;
    	double psi = double(ps) * scal * 2 * CV_PI;
    	Mat krnl = getGaborKernel(Size(wid,wid), sigma, theta, lambda, gamma, psi);
    	Mat fil; filter2D(frame, fil, -1, krnl);
    	Mat krnlRes; resize(krnl, krnlRes, Size(), 10, 10);
    	imshow("kernel", krnlRes);
    	imshow("res", fil);
    	imshow("in", frame);
        Mat draw = frame.clone();
        //ui->show(draw);
        int k = waitKey(40);
        if (k==27) break;
        //if (k=='\t') ui->toggle();
        //if (k=='i') cerr << ui->info() << endl;
    }

    return 0;
}
/*
int main() {
    double a = 10;
    double b = 0.5;
    cv::Mat_<cv::Point2d> x(2, 3);
    double c = -1.3;
    double d = 104;

    std::cerr << "(a, b, c, d) = " << a << ", " << b << ", " << c << ", " << d << ")" << std::endl;
    cv::theRNG().fill(x, cv::RNG::NORMAL, -9, 10);
    std::cerr << "(a, b, c, d) = " << a << ", " << b << ", " << c << ", " << d << ")" << std::endl;

    return 0;
}
*/
