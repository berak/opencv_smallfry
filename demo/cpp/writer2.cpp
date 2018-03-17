#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

int main( int, char** argv )
{
	VideoCapture cap(0);
	VideoWriter writer("my.avi", VideoWriter::fourcc('X','V','I','D'), 30, Size(640,480));
	float all=0;
	Mat frame;
	cap >> frame;   // drop first frames, they're slow
	cap >> frame;
	cap >> frame;
	cap >> frame;
	cap >> frame;
	while(true)
	{
		int64 start = getTickCount();
		cap >> frame;
		writer.write(frame);
		imshow("ocv",frame);
		int64 end = getTickCount();
		float sec = (end-start) / getTickFrequency();
		float t = 0.033 - sec;
		int ticks = t * 1000;
		cerr << sec << "  " << t << " " << ticks << " " << all << endl;
		int k = waitKey(ticks>1?ticks:10);
		if (k==27) break;  // esc. pressed
		all += sec + t;
		if (all>20) break;
	}
    return(0);
}

