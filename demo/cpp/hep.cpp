#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <map>
using namespace cv;
using namespace std;

//
// https://github.com/opencv/opencv_contrib/files/4527696/a-better-first-derivative-approach-for-edge-detection-IJERTV2IS110616.pdf
//
//    A1 A2 A2
// C1 A4 A5 A6 D1
// C2 F  G  H  D2
// C3 B1 B2 B3 D3
//    B4 B5 B6
//
Mat edges(const Mat &img, int nNeighbours=4, float thresh=65.0f) {
    Mat pad, in;
    copyMakeBorder(img,pad,2,2,2,2,BORDER_REFLECT);
    pad.convertTo(in, CV_32F); // so we can pow() it
    int h = in.rows-4;
    int w = in.cols-4;
    Mat A1 = in(Rect(1,0,w,h));
    Mat A2 = in(Rect(2,0,w,h));
    Mat A3 = in(Rect(3,0,w,h));
    Mat A4 = in(Rect(1,1,w,h));
    Mat A5 = in(Rect(2,1,w,h));
    Mat A6 = in(Rect(3,1,w,h));
    Mat B1 = in(Rect(1,3,w,h));
    Mat B2 = in(Rect(2,3,w,h));
    Mat B3 = in(Rect(3,3,w,h));
    Mat B4 = in(Rect(1,4,w,h));
    Mat B5 = in(Rect(2,4,w,h));
    Mat B6 = in(Rect(3,4,w,h));
    Mat C1 = in(Rect(0,1,w,h));
    Mat C2 = in(Rect(0,2,w,h));
    Mat C3 = in(Rect(0,3,w,h));
    Mat D1 = in(Rect(4,1,w,h));
    Mat D2 = in(Rect(4,2,w,h));
    Mat D3 = in(Rect(4,3,w,h));
    Mat F  = in(Rect(1,2,w,h));
    Mat H  = in(Rect(3,2,w,h));

    Mat dr,dc;
    if (nNeighbours == 6) {
        dr = ((B1 + B2 + B3 + B4 + B5 + B6) - (A1 + A2 + A3 + A4 + A5 + A6)) / 6;
        dc = ((A6 + H  + B3 + D1 + D2 + D3) - (A4 + F  + B1 + C1 + C2 + C3)) / 6;
    } else
    if (nNeighbours == 4) {
        dr = ((B1 + B2 + B3 + B5) - (A2 + A4 + A5 + A6)) / 4;
        dc = ((A6 + H  + B3 + D2) - (A4 + F  + B1 + C2)) / 4;
    }

    Mat mag;
    sqrt((dr.mul(dr) + dc.mul(dc))/2, mag);

    return mag > thresh;
}

Mat hep(const Mat &in) {
    int H=in.rows-2;
    int W=in.cols-2;
    Mat cn = in(Rect(1,1,W,H));
    Mat h1 = in(Rect(0,0,W,H));
    Mat h2 = in(Rect(1,0,W,H));
    Mat h3 = in(Rect(2,0,W,H));
    Mat h4 = in(Rect(2,1,W,H));
    Mat h5 = in(Rect(2,2,W,H));
    Mat h6 = in(Rect(1,2,W,H));
    Mat h7 = in(Rect(0,2,W,H));
    Mat h8 = in(Rect(0,1,W,H));

    Mat mask, h(H,W,CV_8U,Scalar(0));
    compare(cn,h8,mask,CMP_GT); bitwise_or(h,1,h,mask);
    compare(cn,h7,mask,CMP_GT); bitwise_or(h,2,h,mask);
    compare(cn,h6,mask,CMP_GT); bitwise_or(h,4,h,mask);
    compare(cn,h5,mask,CMP_GT); bitwise_or(h,8,h,mask);
    compare(cn,h4,mask,CMP_GT); bitwise_or(h,16,h,mask);
    compare(cn,h3,mask,CMP_GT); bitwise_or(h,32,h,mask);
    compare(cn,h2,mask,CMP_GT); bitwise_or(h,64,h,mask);
    compare(cn,h1,mask,CMP_GT); bitwise_or(h,128,h,mask);
    return h;
}

Mat rnoise(const Mat &img) {
    Mat_<uchar> in(img.clone());
    int n = in.total() / 100;
    for (int i=0; i<n; i++) {
        int c = theRNG().uniform(0,in.cols);
        int r = theRNG().uniform(0,in.rows);
        int v = theRNG().uniform(0,2) * 255;
        in(r,c) = v;
    }
    return in;
}

Mat synth(){
    Mat img(400,400,CV_8U,Scalar(255));
    ellipse(img,Point(180,100), Size(110,50), 0, 0, 360, Scalar(120), 40, 8, 0 );
    rectangle(img,Rect(180,240,180,70),Scalar(60),40);
    return img;
}
int main( int argc, char** argv )
{
    //rnoise();
    int N  = 500; // num runs
    //Mat im = imread("battery_t2.jpg",0);
    Mat im = synth();
    Mat rn = rnoise(im);
    Mat d1 = edges(im, 6);
    Mat d2 = edges(rn, 6);
    Mat d  = d2-d1;
    int nz = countNonZero(d);
    cout <<"tot " << d.total() << " nz " << nz << endl;
    Mat dd; multiply(d,d,dd,1,CV_32F);
    double mse = sum(dd)[0] / d.total();
    double mea = sum(d)[0] / d.total();
    cout <<"mea " << mea << " " << sum(d)[0] << " " << (sum(d)[0] / nz) << endl;
    //compare(d1,d2,d,CMP_EQ);
    cout <<"mse " << mse << endl;
    Scalar me,sd;
    meanStdDev(d,me,sd);
    cout <<"dev " << sd[0] << endl;
    cout <<"mea " << me[0] << endl;

/*
    Mat h;
    int64 dt=0;
    for (int i=0; i<N; i++) {
        int64 t0 = getTickCount();
        h = hep(im);
        int64 t1 = getTickCount();
        dt += t1 - t0;
    }
    cout << N << " runs, " << (double(dt) / (N*getTickFrequency())) << endl;
*/
    imshow("I",im);
    imshow("N",rn);
    imshow("D",d);
    imshow("D1",d1);
    imshow("D2",d2);
    waitKey();
    return 0;
}
