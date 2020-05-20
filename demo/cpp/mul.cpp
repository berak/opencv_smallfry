#include <iostream>

#include <opencv2/core.hpp>
using namespace cv;
using namespace std;
int main(int argc, char** argv) {

    Mat a(3,3,CV_8UC3, Scalar(1,2,3));
    cout << a << endl;

    Mat b = a * 2;
    cout << b << endl;

    a *= 2;
    cout << a << endl;

    return 0;
}