#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/bgsegm.hpp"
#include <iostream>
using namespace cv;
using namespace std;

int main() {

Ptr<BackgroundSubtractor> bg1 = bgsegm::createBackgroundSubtractorGSOC(5);
Ptr<BackgroundSubtractor> bg2 = bgsegm::createBackgroundSubtractorGSOC(100);
bool do1=false,do2=false;
	VideoCapture cap(0);
	while(1) {
		Mat f,g;
		cap.read(f);
		cvtColor(f,g,COLOR_BGR2GRAY);
		Mat m1; bg1->apply(g,m1);
		Mat m2; bg2->apply(g,m2);
		Mat b1,b2;
		if (do1) bg1->getBackgroundImage(b1);
		if (do2) bg2->getBackgroundImage(b2);
		//bg2->getBackgroundImage(b2);
		imshow("F",f);
		cout << do1 << " " << do2 << endl;
		int k = waitKey(5);
		if (k==27) {
			return 0;
		}
		if (k=='1') do1=!do1;
		if (k=='2') do2=!do2;
	}
	return 0;
}