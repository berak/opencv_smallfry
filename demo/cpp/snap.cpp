#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
using namespace cv;
using namespace std;

int main() {
	VideoCapture cap(0);
	while(1) {
		Mat f;
		cap.read(f);
		imshow("F",f);
		int k = waitKey(5);
		if (k==' ') {
			imwrite("snap.png",f);
			return 0;
		}
	}
	return 0;
}