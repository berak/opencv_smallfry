#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>
#include <deque>
#include "profile.h"

using namespace cv;
using namespace std;

// http://www.scholarpedia.org/article/Local_Binary_Patterns

int NZ = 48;
int BS = 24;

static const vector<Point> samps { // fixed, tl() landmarks
        	      Point(22,8),  Point(50,8),
	Point(12,24), Point(24,20), Point(48,20), Point(60,24),
    Point(16,40), Point(22,40), Point(50,40), Point(54,40),
        	  Point(16,60), Point(36,60), Point(54,60),
        	      Point(22,70), Point(50,70)
};

static const int uniform[256] = { // uniform2 pattern lookup
    0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,
    14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,
    58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
    58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,
    58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,
    58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,
    58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
    58,35,36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,
    58,58,58,58,58,58,41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,
    58,58,58,50,51,52,58,53,54,55,56,57
};


static const int off_9[] = { // neighbours
    -1,-1, 0,-1, 1,-1,
    -1, 0,       1, 0,
    -1, 1, 0, 1, 1, 1,
};

typedef deque< Mat_<uchar> > Sequence;

void lbp_xy(const Sequence &seq, Mat &hist, Point p) {
    Mat_<float> H(1,59);
    const int m = 1, fixed = seq.size() / 2;
    for (int c=p.x+m; c<p.x+BS-m; c++) {
        for (int r=p.y+m; r<p.y+BS-m; r++) {
            uchar v = 0;
            uchar cen = seq[fixed](r,c);
	        for (int o=0; o<8; o++)
	            v |= (seq[fixed](r + off_9[o*2+1], c + off_9[o*2]) > cen) << o;
            H(uniform[v])++;
        }
    }
    normalize(H,H);
    hist.push_back(H);
}

void lbp_xz(const Sequence &seq, Mat &hist, Point p) {
    Mat_<float> H(1,59);
    const int m = 1, fixed = seq[0].rows / 2;
    for (int c=p.x+m; c<p.x+BS-m; c++) {
        for (int z=m; z<BS-m; z++) {
            uchar v = 0;
            uchar cen = seq[z](fixed,c);
	        for (int o=0; o<8; o++)
	            v |= (seq[z + off_9[o*2+1]](fixed, c + off_9[o*2]) > cen) << o;
            H(uniform[v])++;
        }
    }
    normalize(H,H);
    hist.push_back(H);
}

void lbp_yz(const Sequence &seq, Mat &hist, Point p) {
    Mat_<float> H(1,59);
    const int m = 1, fixed = seq[0].cols / 2;
    for (int r=p.y+m; r<p.y+BS-m; r++) {
        for (int z=m; z<BS-m; z++) {
            uchar v = 0;
            uchar cen = seq[z](r,fixed);
	        for (int o=0; o<8; o++)
	            v |= (seq[z + off_9[o*2]](r + off_9[o*2+1], fixed) > cen) << o;
            H(uniform[v])++;
        }
    }
    normalize(H,H);
    hist.push_back(H);
}


Mat lbptop(const Sequence &seq, const vector<Point> &samps) {
	PROFILE;
	Mat hist;
	for (Point p:samps) {
		lbp_xy(seq, hist, p);
		lbp_xz(seq, hist, p);
		lbp_yz(seq, hist, p);
	}
	return hist.reshape(1,1);
}

Mat img_yz(const Sequence &seq, Point p) {
	Mat r(BS, BS, CV_8U);
	int x = p.x + BS/2;
	for (int z=0; z<seq.size(); z++) {
		for (int y=0; y<BS; y++) {
			r.at<uchar>(y,z) = seq[z].at<uchar>(y+p.y,x);
		}
	}
	return r;
}
Mat img_xz(const Sequence &seq, Point p) {
	Mat r(BS, BS, CV_8U);
	int y = p.y + BS/2;
	for (int z=0; z<seq.size(); z++) {
		for (int x=0; x<BS; x++) {
			r.at<uchar>(x,z) = seq[z].at<uchar>(y,x+p.x);
		}
	}
	return r;
}


#include "Mosse.h"

int main(int argc, const char* argv[]) {
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
	    		Mat hist = lbptop(seq, samps);
	    	}
	    	Mat viz; // the xy,xz,yz planes, per sample point
	  		Mat fbgr; resize(frame(box),fbgr,Size(96,96));
	    	for (Point p: samps) {
	    		rectangle(fbgr, p, p+Point(BS,BS), Scalar(0,0,200), 1);
	    		Mat xy = det(Rect(p.x,p.y,BS,BS)).clone();
	    		hconcat(xy,img_xz(seq,p),xy);
	    		hconcat(xy,img_yz(seq,p),xy);
	    		if (viz.empty()) viz=xy;
	    		else vconcat(viz,xy,viz);
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

