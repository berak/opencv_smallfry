#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
	int N = 4; // num cams:
	Size S(320,240);
	vector<VideoCapture> caps(N);

	for (int i=0; i<N; i++)
	{
		if ( ! caps[i].open(i) )
		{
			cerr << "capture " << i << " won't open !" << endl;
			return -1;
		}
		cerr << "cap " << i << " : " << caps[i].isOpened() << endl;

		// lower resolution:
		caps[i].set(CAP_PROP_FRAME_WIDTH,  S.width);
		caps[i].set(CAP_PROP_FRAME_HEIGHT, S.height);
	}

	while(1)
	{
		for (int i=0; i<N; i++)
		{
			Mat frame;
			if ( ! caps[i].read(frame) )
			{
				cerr << "could not read image from capture " << i << endl;
				return -1;
			}

			//
			// process frame from capture i ...
			//
			// show it:
			imshow(format("cap%d",i), frame);
		}
		if (waitKey(10)==27) break;
	}
}
