#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>
#include <deque>
#include "profile.h"
#include "lbptop.h"

using namespace cv;
using namespace std;


// -----8<-----------------------------------------------------------
static const vector<Point> samps_face { // 15 fixed, tl() landmarks in (96,96) space
        	      Point(22,8),  Point(50,8),
	Point(12,24), Point(24,20), Point(48,20), Point(60,24),
    Point(16,40), Point(22,40), Point(50,40), Point(54,40),
        	  Point(16,60), Point(36,60), Point(54,60),
        	      Point(22,70), Point(50,70)
};

// -------8<------------------------------------------------------------------
#include "Mosse.h"

int main(int argc, const char* argv[]) {
	int BS = lbp_get_blocksize();
	namedWindow("planes", 0);
	namedWindow("samps", 0);
	VideoCapture cap(0);
	CascadeClassifier cad("haarcascade_frontalface_alt.xml");
	Ptr<ml::SVM> svm = ml::SVM::create();
	Sequence seq;
	Mosse mosse;
	while(cap.isOpened()) {
		PROFILEX("per_frame")
		Mat frame, gray;
		cap >> frame;
	    cv::resize(frame,frame,cv::Size(frame.cols/2, frame.rows/2));
	    cv::cvtColor(frame,gray,CV_BGR2GRAY);
	    cv::equalizeHist(gray,gray);

		Rect box, screen(0, 0, frame.cols, frame.rows);
	    {
	    	PROFILEX("mosse.update")
	    	mosse.update(gray,box);
    	}
    	if (box.empty()) {
    		PROFILEX("detect")
			vector<Rect> faces;
	   		cad.detectMultiScale(gray,faces,1.1,4,CV_HAAR_FIND_BIGGEST_OBJECT,cv::Size(30,30));
		    if (faces.size()) {
		    	box = faces[0];
		    }
    	}
    	box &= screen;
	    if (! box.empty()) {
	    	PROFILEX("per_box")
	    	rectangle(frame, box, Scalar(0,200,0), 1);
	    	Mat det = gray(box);
	    	resize(det,det,Size(96,96));
	    	seq.push_back(det);
	    	if (seq.size() > BS) {
	    		seq.pop_front();
	    		Mat hist = lbptop(seq, samps_face);
	    	}
	    	Mat viz; // the xy,xz,yz planes, per sample point
	  		Mat fbgr;
	  		{ PROFILEX("vizu")
	  		resize(frame(box),fbgr,Size(96,96));
	    	for (Point p: samps_face) {
	    		rectangle(fbgr, p, p+Point(BS,BS), Scalar(0,0,200), 1);
	    		Mat xy = det(Rect(p.x,p.y,BS,BS)).clone();
	    		hconcat(xy,img_xz(seq,p),xy);
	    		hconcat(xy,img_yz(seq,p),xy);
	    		if (viz.empty()) viz=xy;
	    		else vconcat(viz,xy,viz);
	    	}
	    	}
    		imshow("planes", viz);
    		imshow("samps", fbgr);
	    }
		imshow("TOP",frame);
		int k = waitKey(10);
		switch(k) {
			case 27: return 0;
			case ' ': if (!box.empty()) {
	    		PROFILEX("mosse.init")
				mosse.init(gray,box);
				break;
			}
			case 'r':
			case 'c': {
				mosse.clear();
				break;
			}
		}
	}
  	return 0;
}

