#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>
#include <deque>
#include "profile.h"
#include "lbptop.h"

using namespace cv;
using namespace std;

// http://www.scholarpedia.org/article/Local_Binary_Patterns

int NB = 4;  // n blocks in x and y

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


void lbp_xy(const Sequence &seq, Mat &hist, const Rect &rec) {
    Mat_<float> H(1,59, 0.0f);
    const int m = 1, fixed = seq.size() / 2;
    for (int c=rec.x+m; c<rec.x+rec.width-m; c++) {
        for (int r=rec.y+m; r<rec.y+rec.height-m; r++) {
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
void lbp_xz(const Sequence &seq, Mat &hist, const Rect &rec) {
    Mat_<float> H(1,59, 0.0f);
    const int m = 1, fixed = seq[0].rows / 2;
    for (int c=rec.x+m; c<rec.x+rec.width-m; c++) {
        for (int z=m; z<int(seq.size())-m; z++) {
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
void lbp_yz(const Sequence &seq, Mat &hist, const Rect &rec) {
    Mat_<float> H(1,59,0.0f);
    const int m = 1, fixed = seq[0].cols / 2;
    for (int r=rec.y+m; r<rec.y+rec.height-m; r++) {
        for (int z=m; z<int(seq.size())-m; z++) {
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



Mat lbptop(const Sequence &seq) {
	PROFILE;
    int w = seq[0].cols / NB;
    int h = seq[0].rows / NB;
	Mat hist;
    for (int i=0;i<NB; i++) {
    	for (int j=0;j<NB; j++) {
    		Rect r(j*h, i*w, w, h);
            r &= Rect(0,0,seq[0].cols, seq[0].rows);
            lbp_xy(seq, hist, r);
    		lbp_xz(seq, hist, r);
    		lbp_yz(seq, hist, r);
    	}
    }
	return hist.reshape(1,1);
}


Mat img_yz(const Sequence &seq, const Rect &r) {
	Mat im(r.height, r.width, CV_8U);
	int x = r.x + r.width/2;
	for (int z=0; z<seq.size(); z++) {
		for (int y=0; y<r.height; y++) {
			im.at<uchar>(y,z) = seq[z].at<uchar>(y+r.y,x);
		}
	}
	return im;
}

Mat img_xz(const Sequence &seq, const Rect &r) {
    Mat im(r.height, r.width, CV_8U);
	int y = r.y + r.height/2;
	for (int z=0; z<seq.size(); z++) {
		for (int x=0; x<r.width; x++) {
			im.at<uchar>(x,z) = seq[z].at<uchar>(y,x+r.x);
		}
	}
	return im;
}

int lbpFlow(const String &filename, Mat &desc, int frameFrom, int frameTo) {
    VideoCapture cap(filename);
    if ( !cap.isOpened() )
        return -1;

    cap.set(CAP_PROP_POS_FRAMES, frameFrom);

    Mat frame,gray;
    Sequence seq;

    for (;;) {
        if ((frameTo != 0) && ((int)cap.get(CAP_PROP_POS_FRAMES) == frameTo))
            break;

        cap >> frame;
        if (frame.empty())
            break;

        resize(frame,frame,Size(320,240));
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray,gray);
        seq.push_back(gray);
    }
    // repeat for shorter vids !
    for (int i=0; seq.size()<(frameTo-frameFrom); i++) seq.push_back(seq[i]);
    //cout << seq.size() << seq[0].size() << endl;

    desc = lbptop(seq);
    return 0;
}
