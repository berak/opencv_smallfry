#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

int main()
{
	Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();
	VideoCapture capture(0);

	UMat stat_backgnd;
	capture.read(stat_backgnd);

	for(;;){

    	UMat small_raw_frame, fgmask;
		capture.read(small_raw_frame);

		pMOG2->apply(small_raw_frame, fgmask);

		int cnt =countNonZero(fgmask);
		if ( cnt<= 5000 ) stat_backgnd = small_raw_frame;

		imshow("fg", fgmask);
		imshow("back", stat_backgnd);
		if (waitKey(10) == 27) break;
	}
	return 0;
}
