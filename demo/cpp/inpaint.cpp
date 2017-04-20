#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>
#include <iostream>
using namespace cv;
using namespace std;


void inpaint_rgb() {
    string path_mask = "mask1.png";
    string path_gt = "1.png";
    string path_img = path_gt;
    string path_inpaint = "result-1.png";

    Mat gt = cv::imread(path_gt, cv::IMREAD_GRAYSCALE);
    Mat mask = cv::imread(path_mask, cv::IMREAD_GRAYSCALE);
    Mat img = cv::imread(path_img, cv::IMREAD_COLOR);

    img.setTo(cv::Scalar(0,0,0), mask == 255);
    Mat inpainted;
    mask = 255 - mask;
    cv::xphoto::inpaint(img, mask, inpainted, 0);
    cv::imwrite(path_inpaint, inpainted);
}

int main()
{
    inpaint_rgb();
    return 0;
}
