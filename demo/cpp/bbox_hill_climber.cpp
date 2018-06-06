#include "opencv2/opencv.hpp"
#include <fstream>

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
 	Mat im = imread("polkadots.png");
 	Mat gr; cvtColor(im,gr,COLOR_BGR2GRAY);
 	erode(gr,gr,Mat());
 	erode(gr,gr,Mat());
 	Rect best, bounds(0,0,im.cols,im.rows);
 	float best_ratio=0;
 	for (int i=0; i<10000; i++) {
 		Rect r(theRNG().uniform(0,im.cols/4),
 			   theRNG().uniform(0,im.rows/4),
 			   theRNG().uniform(3*im.cols/4,im.cols),
 			   theRNG().uniform(3*im.rows/4,im.rows));
 		r &= bounds;
 		int n = countNonZero(gr(r));
 		int N = r.width*r.height;
 		float ratio = float(n)/N;
 		if (ratio > best_ratio && (r.width*r.height) > (best.width*best.height)) {
 			best = r;
 			best_ratio = ratio;
 		}
 	}
 	rectangle(im,best,Scalar(0,0,200),3);
 	cout << best_ratio << " " << best << endl;
 	imshow("best",im);
 	imshow("gr",gr);
 	waitKey();
    return 0;
}
