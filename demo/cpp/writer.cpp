#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

int main( int, char** argv )
{
	VideoWriter writer("my.avi", VideoWriter::fourcc('M','J','P','G'), 25, Size(640,480));
	for (int i=0; i<50000; i++) {
		Mat frame(480,640,CV_8UC3, Scalar(100,100,100));
		putText(frame,format(";) %d",i), Point(200,200), FONT_HERSHEY_PLAIN, 5, Scalar(255,0,0), 3);
		writer.write(frame);
		imshow("W",frame);
		waitKey(150);
		cout << i << "\r";
	}
    return(0);
}

