#include "opencv2/opencv.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;

#include <iostream>
using namespace std;

float sim(const Mat &q, const Mat &t) {
	Mat res;
	matchTemplate(q,t,res,TM_CCOEFF);
	double m; Point p;
	minMaxLoc(res,0,&m,0,&p);
	// p should be in top-left half of our quadrant
	cout << p << " " << (Rect(0,0,q.cols/2,q.rows/2).contains(p)) << " " << m << endl;
	return m;
}
int main()
{
	Mat input = imread("battery2.jpg", 0);
	Mat templ = imread("battery_t2.jpg", 0);

	// slice image into 4 overlapping regions:
	Rect tl(0,0,input.cols/1.9,input.rows/1.9);
	Rect tr(input.cols/2.3,0,input.cols/1.9,input.rows/1.9);
	Rect bl(0,input.rows/2.3,input.cols/1.9,input.rows/1.9);
	Rect br(input.cols/2.3,input.rows/2.3,input.cols/1.9,input.rows/1.9);
	// check similarity per region:
	float r1 = sim(input(tl),templ);
	float r2 = sim(input(tr),templ);
	float r3 = sim(input(bl),templ);
	float r4 = sim(input(br),templ);
	return 0;
}


/*
void bd() {
    SimpleBlobDetector::Params prm;
    // This is default parameters for SimpleBlobDetector
    prm.thresholdStep = 10;
    prm.minThreshold = 5;
    prm.maxThreshold = 520;
    prm.minRepeatability = 2;
    prm.minDistBetweenBlobs = 200;
    prm.filterByColor = false;
    prm.blobColor = 0;
    prm.filterByArea = true;
    prm.minArea = 50;
    prm.maxArea = 5000;
    prm.filterByCircularity = true;
    prm.minCircularity = 0.9f;
    prm.maxCircularity = (float)1e37;
    prm.filterByInertia = false;
    prm.minInertiaRatio = 0.1f;
    prm.maxInertiaRatio = (float)1e37;
    prm.filterByConvexity = false;
    prm.minConvexity = 0.95f;
    prm.maxConvexity = (float)1e37;


    Ptr<SimpleBlobDetector> b = SimpleBlobDetector::create(prm);
    vector<KeyPoint>  keyImg;
    b->detect(input, keyImg, Mat());
    cout << keyImg.size() << endl;
    for (auto k: keyImg) {
    	circle(input, k.pt, 10, Scalar(0,0,200), 5);
    }
}*/