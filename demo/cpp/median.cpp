#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, const char * argv[])
{
    cv::Mat a = cv::Mat::zeros(100, 200, 5);
    cv::Mat b;
    cv::medianBlur(a, b, 5);
}