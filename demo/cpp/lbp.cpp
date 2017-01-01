
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace cv::ml;
using namespace std;

Mat lbp1(const Mat &img) {
	Mat lbp(img.size(), CV_8U);
	for (size_t i=1; i<img.rows-1; i++) {
		for (size_t j=1; j<img.cols-1; j++) {
			int v = 0;
			int c = img.at<uchar>(i,j);
			v += (c < img.at<uchar>(i-1,j-1));
			v += (c < img.at<uchar>(i-1,j  ))*2;
			v += (c < img.at<uchar>(i-1,j+1))*4;
			v += (c < img.at<uchar>(i  ,j+1))*8;
			v += (c < img.at<uchar>(i+1,j+1))*16;
			v += (c < img.at<uchar>(i+1,j  ))*32;
			v += (c < img.at<uchar>(i+1,j-1))*64;
			v += (c < img.at<uchar>(i  ,j-1))*128;
			lbp.at<uchar>(i,j) = v;
		}
	}
	return lbp;
}

Mat lbp2(const Mat &im) {
	Mat_<uchar> img(im);
	Mat_<uchar> lbp(img.size());
	for (size_t i=1; i<img.rows-1; i++) {
		for (size_t j=1; j<img.cols-1; j++) {
			int v = 0;
			int c = img(i,j);
			v += (c < img(i-1,j-1));
			v += (c < img(i-1,j  ))*2;
			v += (c < img(i-1,j+1))*4;
			v += (c < img(i  ,j+1))*8;
			v += (c < img(i+1,j+1))*16;
			v += (c < img(i+1,j  ))*32;
			v += (c < img(i+1,j-1))*64;
			v += (c < img(i  ,j-1))*128;
			lbp(i,j) = v;
		}
	}
	return lbp;
}

int main(int argc, char** argv) {
    return 0;
}
