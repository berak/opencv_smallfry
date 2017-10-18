#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>
#include <deque>
#include "profile.h"
#include "lbptop.h"

using namespace cv;
using namespace std;


// -------8<------------------------------------------------------------------
#include "Mosse.h"

int main(int argc, const char* argv[]) {
	int BS = 24;
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
	    		Mat hist = lbptop(seq);
	    	}
	    	Mat viz; // the xy,xz,yz planes, per sample point
	  		{
	  			PROFILEX("vizu")
		  		int NB = 4;
		  	    int w = seq[0].cols / NB;
			    int h = seq[0].rows / NB;
			    for (int i=0;i<NB; i++) {
			    	for (int j=0;j<NB; j++) {
			    		Rect r(j*h, i*w, w, h);
			    		Mat xy = det(r).clone();
			    		hconcat(xy,img_xz(seq,r),xy);
			    		hconcat(xy,img_yz(seq,r),xy);
		    			if (viz.empty()) viz=xy;
		    			else vconcat(viz,xy,viz);
		    		}
		    	}
	    	}
    		imshow("planes", viz);
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

