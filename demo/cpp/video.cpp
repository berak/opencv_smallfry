#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

int main()
{
	VideoCapture cap("C:/p/ocv/opencv_smallfry/mace/cv/tracking/faceocc2/data/faceocc2.webm");
	if (! cap.isOpened())
	{
		cout << "could not open the VideoCapture !" << endl;
		return -1;
	}
	while(true)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty()) // movie is over
		{
			break;
		}
		imshow("ocv",frame);
		int k = waitKey(10);
		if (k==27) break;  // esc. pressed
	}
	return 0;
}
