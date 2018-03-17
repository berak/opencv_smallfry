#include <iostream>
#include "opencv2/core/ocl.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"
#include <opencv2/core/utils/trace.hpp>

using namespace cv;
using namespace cv::face;
using namespace std;

void foo(InputArray in) {
    Mat m = in.getMat();
    cvtColor(in,m,COLOR_BGR2GRAY);
}
int main() {
    Mat m(3,3,CV_8UC3,Scalar(1,2,3));
    cerr << m << endl;
    foo(m);
    cerr << m << endl;
     return 0;
}
