#include <iostream>

#include <opencv2/core.hpp>
using namespace cv;
using namespace std;
int main(int argc, char** argv) {
    Mat I = Mat::eye(3,3,CV_32F);
    Mat_<float> r(1,3); r << 0.1,0,0;
    Mat v;
    Rodrigues(r,v);
    cout << v << endl;
    Mat m2;
    Mat n = m1/m2;
    cout << n << endl;
/*
    divide(1,m1,n);
    cout << n << endl;

    n = m1 & m2;
    cout << n << endl;
    */
    return 0;
}