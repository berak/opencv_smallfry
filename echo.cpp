#include "opencv/cv.h"
#include "opencv/highgui.h"
using namespace cv;

#include <vector>

struct EchoChamber
{
	struct Tap 
	{
		int pos;
		float weight;
		Tap( int p=0, float w=0 ) : pos(p), weight(w) {}
	};
	std::deque<Mat> ring;
	Tap taps[8];
	int len;

	void setTap(int t,  int p, float w )
	{
		taps[t].pos = p;
		taps[t].weight = w;
		len = 0;
		for ( size_t i=0; i<8; i++ )
		{
			int pz = taps[i].pos;
			if ( !pz ) break;
			len = max(len,pz+1);
		}
	}

	Mat mix( const Mat & img, float fac )
	{
		ring.push_front(img);
		if ( ring.size() < len )
			return img;

		while ( ring.size() > len )
		{
			ring.pop_back();
		}

		Mat res,tmp;
		multiply( fac, img, res );

		// now add the weighted taps
		for ( size_t i=0; i<8; i++ )
		{
			int idx = taps[i].pos;
			if ( idx==0 ) break;
			multiply(taps[i].weight, ring[idx], tmp);
			add( tmp , res, res );
		}
		return res;
	}
};


EchoChamber echo;

void t0(int p, void *) { echo.setTap(0,p,0.33); }
void t1(int p, void *) { echo.setTap(1,p,0.33); }

int main()
{
	//echo.setTap(0,10,0.33f);
	//echo.setTap(1,20,0.33f);
	//namedWindow("video", 1);
	namedWindow("echo", 0);
	int a=0, b=0;
	createTrackbar( "t 0", "echo", &a, 100, t0 );
	createTrackbar( "t 1", "echo", &b, 100, t1 );
    
	VideoCapture cap;
	cap.open(0);
	Mat frame; 
	Mat processed;
	for(;;) {
		cap >> frame;
		if(!frame.data) break;

		processed = echo.mix(frame.clone(), 0.33f);

		imshow("echo", processed);
		if(waitKey(30) >= 0) break;
	}	
	return 0;
}


