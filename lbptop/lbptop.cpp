#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>
#include <deque>
#include "profile.h"
#include "lbptop.h"

using namespace cv;
using namespace std;

// http://www.scholarpedia.org/article/Local_Binary_Patterns

int NZ = 48;
int BS = 24; // override for hockey

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

void lbp_set_blocksize(int s)
{
	BS = s;
}
int lbp_get_blocksize()
{
	return BS;
}


void lbp_xy(const Sequence &seq, Mat &hist, Point p) {
    Mat_<float> H(1,59, 0.0f);
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
    Mat_<float> H(1,59, 0.0f);
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
    Mat_<float> H(1,59,0.0f);
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
