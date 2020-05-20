#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void Cayley(const Mat &A, Mat &R) {
    const static Mat I = Mat::eye(3,3,CV_32F);
    cout << A << endl;
    Mat i0 = I - A;
    subtract(I,A,i0);
    Mat i1 = I + A;
    cout << i0 << endl;
    cout << i1 << endl;
    //Mat i2; invert(i0, i2);
    //Mat i2; transpose(i0, i2);
    ///cout << i2 << endl;
    gemm(i0,i1,1,noArray(),0,R,GEMM_1_T);
    R = I - R;
    cout << R << endl;
    cout << R.diag() << endl;
}
Mat sqewt(const Mat &a) {
    int d=std::max(a.rows, a.cols);
    Mat r(d,d,CV_32F,0.0f);
    for (int i=0; i<d; i++) {
        r.at<float>(i,i) = a.at<float>(i);
    }
    return r;
}
int main(int argc, char** argv) {
    Mat_<float> r(1,3); r << 0.1,0,0;
    Mat m,v,c;
    Rodrigues(r,m);
    cout << m << endl;
    Rodrigues(m,v);
    cout << v << endl;
    cout << "-----------------" << endl;
    //Mat_<float> C(3,3); C << 0.1,0,0, 0,1,0, 0,1,0;
    Mat s = sqewt(r);
    Cayley(s,c);
    return 0;
}