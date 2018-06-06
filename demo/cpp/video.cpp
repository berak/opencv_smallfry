#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

int main()
{
	VideoCapture cap("my.avi");
	if (! cap.isOpened())
	{
		cout << "could not open the VideoCapture !" << endl;
		return -1;
	}
	int f = 0;
	while(true)
	{
		cout << f << endl;
		Mat frame;
		bool ok = cap.read(frame);
		if (ok && !frame.empty()) // movie is over
		{
			imshow("ocv",frame);
			f++;
		} else
		{
			//cap.release();
			cap.open("my.avi");
			bool ok = cap.set(CAP_PROP_POS_FRAMES, f);
			cout << ok << " " << f << " " << cap.get(CAP_PROP_POS_FRAMES) << endl;
		}
		int k = waitKey(10);
		if (k==27) break;  // esc. pressed
	}
	return 0;
}
