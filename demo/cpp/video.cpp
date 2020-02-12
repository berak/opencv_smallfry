#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

int main(int argc, char**argv)
{
	int id = argc>1 ? atoi(argv[1]) : 0;
	VideoCapture cap(id);
	namedWindow("ocv", 0);
	if (! cap.isOpened())
	{
		cout << "could not open the VideoCapture !" << endl;
		return -1;
	}
	Mat frame;
	while(cap.read(frame))
	{
		imshow("ocv",frame);
		int k = waitKey(10);
		if (k==27) break;  // esc. pressed
		if (k=='i') cap.set(CAP_PROP_SETTINGS,1);
	}
	return 0;
}
