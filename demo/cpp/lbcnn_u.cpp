#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
using namespace cv;
using namespace std;

#include "profile.h"


struct layer {
	UMat W;
	vector<Point> pts[2];
	Size sz;
	int m,w;

	layer(const Size &s, int m, int w) : m(m),w(w) {
	    sz = Size(s.width - w, s.height - w);
	    W = UMat(m, sz.area(), CV_32F, 0.0f);
	    randu(W, 0.0, 0.25);
	    pts[0].resize(m);
	    pts[1].resize(m);
	    RNG r(theRNG());//(getTickCount());
        for (int i=0; i<m; i++) {
	    	pts[0][i] = Point(r.uniform(0,w), r.uniform(0,w));
		    pts[1][i] = Point(r.uniform(0,w), r.uniform(0,w));
		}
    	cerr << "W" << W.size() << W.total() << endl;
	}
	UMat conv(const UMat &I, Point p, Point q) const {
	    PROFILE;
	    int M = I.rows;
	    int N = I.cols;
	    UMat ma,mb;
	    ma = I(Range(p.y,p.y+M-w), Range(p.x,p.x+N-w));
	    mb = I(Range(q.y,q.y+M-w), Range(q.x,q.x+N-w));
		{ PROFILEX("sub");
		UMat sub;
		subtract(ma,mb,sub);
		return sub;
		}
	}
	UMat forward(const UMat &A) const {
	    PROFILE;
	    UMat S(sz, CV_32F, 0.0f);
	    for (int i=0; i<m; i++) {
	    	UMat c = conv(A, pts[0][i], pts[1][i]);
	 	    {
			    PROFILEX("mul")
		 	    //multiply(wr, cr, cr, CV_32F);
			    c.reshape(1,1).mul(W.row(i));
	 	    }

		    {
			    PROFILEX("add")
			    add(S,c,S);
			}
		}
		return S;
	}
	Size outsize() const { return sz; }
};

int main(int argc, char **argv) {
	bool useOCL = argc > 1;
    ocl::setUseOpenCL(useOCL);
    cout << "ocl " << cv::ocl::useOpenCL() << endl;

    Mat A = imread("face.png", 0);
    UMat I,u1,u2;
    A.copyTo(u1);
    resize(u1,u2,Size(32,32));
    u2.convertTo(I,CV_32F,1.0/255);

    cerr << "A" << I.size() << I.total() << endl;
    layer l1(I.size(), 128, 5);
    layer l2(l1.outsize(), 64,3);
    layer l3(l2.outsize(), 32,3);

    int maxgen = useOCL ? 50 : 500;
	for (int i=0; i<maxgen; i++) {
		PROFILEX("gen")
	    UMat B = l1.forward(I);
	    UMat C = l2.forward(B);
	    UMat D = l3.forward(C);
	}
	return 0;
}
