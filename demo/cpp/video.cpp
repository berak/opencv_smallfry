#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

int main()
{
	VideoCapture cap("http://www.youtube.com/get_video?&video_id=pSijg_QCgYk");
	while(cap.isOpened())
	{
		Mat f;
		cap >> f;
		imshow("ocv",f);
		int k = waitKey(10);
		if (k==27) break;
	}
	return 0;
}
