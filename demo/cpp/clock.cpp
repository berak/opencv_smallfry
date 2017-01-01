
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

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

#define HAVE_CPP11 (__cplusplus >= 201103L || ((defined(_MSC_VER) && _MSC_VER >= 1800)))

int64 steady_ticks() {
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    auto r = (now.time_since_epoch()).count();
cout << "T " << sizeof(r) << endl;
    return r;
    //return std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
}
int main(int argc, char** argv) {
	cout << __cplusplus << endl;
	//cout << HAVE_CPP11 << endl;
#if HAVE_CPP11
	cout << "YES." << endl;
#endif
#ifdef HAVE_CPP11
	cout << "OK." << endl;
#endif
	cout << "int64 " << sizeof(int64) << endl;

	int64 t0 = steady_ticks();
    int64 t1 = getTickCount();
	cerr << t0 << endl;
	cerr << t1 << endl << endl;
	Mat img(500,500,CV_8U);
	for (int i=0; i<1000;  i++)
		lbp2(img);
	int64 t2 = steady_ticks();
    int64 t3 = getTickCount();
	cerr << t2 << endl;
	cerr << t3 << endl << endl;
	double t4 = double(t2-t0) / 1e9; // nano
	double t5 = double(t3-t1) / getTickFrequency();
	cerr << t4 << endl;
	cerr << t5 << endl;
    return 0;
}
