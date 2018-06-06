#include <opencv2/opencv.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/core/cv_cpu_helper.h>

#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <iostream>
#include "profile.h"

using namespace std;
using namespace cv;
#include <chrono>
#include <thread>

using namespace cv;
using namespace std;

struct Succubus {
	const int BLOCKSIZE=4;
	deque<Mat_<float>> past;

	// 4x4x4
	Mat grad_xy() {
		PROFILE;
		//Mat grad = past[0] - past[1] + past[2] - past[3];

		Mat_<float> grad = Mat_<float>::zeros(4,4);
		for (int x=0; x<BLOCKSIZE; x++) {
			for (int y=0; y<BLOCKSIZE; y++) {
				grad(y,x) += past[0](y,x);
				grad(y,x) -= past[1](y,x);
				grad(y,x) += past[2](y,x);
				grad(y,x) -= past[3](y,x);
			}
		}
		return grad;
	}
	// 4x4x4
	Mat grad_xz() {
		PROFILE;
		Mat_<float> grad = Mat_<float>::zeros(4,4);
		for (int x=0; x<4; x++) {
			for (int y=0; y<4; y++) {
				grad(y,x) += past[y](0,x);
				grad(y,x) -= past[y](1,x);
				grad(y,x) += past[y](2,x);
				grad(y,x) -= past[y](3,x);
			}
		}
		return grad;
	}
	// 4x4x4
	Mat grad_yz() {
		PROFILE;
		Mat_<float> grad = Mat_<float>::zeros(4,4);
		for (int x=0; x<4; x++) {
			for (int y=0; y<4; y++) {
				float &g = grad(y,x);
				g -= past[x](y,0);
				g += past[x](y,1);
				g -= past[x](y,2);
				g += past[x](y,3);
			}
		}
		return grad*.1;
	}
	Mat hist(const Mat_<float> &grad) {
		PROFILE;
		Mat_<float> his = Mat_<float>::zeros(1,8);
		for (int x=0; x<4; x++) {
			for (int y=0; y<4; y++) {
				float v = abs(grad(y,x));
				//int s = int(sqrt(v/16));
				int s = int(v*0.5);
				s = s>7 ? 7 : s;
				his(0,s) ++;
			}
		}
		// normalize
		float m = 0;
		for (int i=0; i<8; i++) m = std::max(m, his(0,i));
		for (int i=0; i<8; i++) his(0,i) /= m;

		return his;
	}
	Mat sift() {
		PROFILE;
		Mat g1 = (grad_xy());
		Mat g2 = (grad_xz());
		Mat g3 = (grad_yz());
		Mat d1 = hist(g1);
		Mat d2 = hist(g2);
		Mat d3 = hist(g3);
		vconcat(d1, d2, d1);
		vconcat(d1, d3, d1);

		/*
		hconcat(g1,g2,g1);
		hconcat(g1,g3,g1);
		resize(g1,g1,Size(3*128,128));
		*/
		Mat ch[] = {g1,g2,g3};
		Mat gm; merge(ch, 3, gm);
		resize(gm,gm,Size(128,128));
		imshow("SIFT", gm*0.05);

		return d1.reshape(1,1);
	}
};

Mat cube(const Mat &in) {
	PROFILE;

}

int main( int argc, char** argv )
{
	VideoCapture cap(0);
	Succubus sub;
	Mat m;
	while(cap.read(m)) {
		Rect r(400,400,4,4);
		Mat fm; m(r).convertTo(fm, CV_32F);
		sub.past.push_back(fm);
		if (sub.past.size() > 4)
			sub.past.pop_front();
		if (sub.past.size() == 4) {
			Mat_<float> s = sub.sift();
		//cout << s.reshape(1,3) << endl;
			for (int i=0; i<24; i++) {
				int so = s(0,i) * 100;
				rectangle(m, Rect(200+i*10,200-so,10,so), Scalar(0,2*so,0),1);
			}
		}
		rectangle(m, r, Scalar(200,0,0),1);
		imshow("lalala", m);
		int k = waitKeyEx(10);
		if (k==27)
			break;
	}
  return 0;
}
