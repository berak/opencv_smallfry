#include <iostream>
#include <deque>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

double score(const Mat &m, const Mat &n) {
	Mat res;
	matchTemplate(m,n,res,TM_CCOEFF_NORMED);
	double M;
	minMaxLoc(res,0,&M,0,0);
}

void split(const Rect &a, Rect &b, Rect &c) {
	if (a.height>a.width) {
		b = Rect(a.x, a.y, a.width, a.height/2);
		c = Rect(a.x, a.y+a.height/2, a.width, a.height/2);
		cerr << "vert ";
	} else {
		b = Rect(a.x, a.y, a.width/2, a.height);
		c = Rect(a.x+a.width/2, a.y, a.width/2, a.height);
		cerr << "horz ";
	}
}

struct rect { Rect r; double s; };
bool sort_score(const rect &a, const rect &b) { return a.s > b.s; }

int main() {
	Mat img = imread("dots.png");
	Mat tmp = img(Rect(31,26,25,26));

	deque<rect> bp;
	bp.push_back(rect{Rect(0,0,img.cols,img.rows),0});
	for (int g=0; g<500 && !bp.empty(); g++) {
		double s = bp.back().s;
		Rect r1,r2, o = bp.back().r;
		split(o,r1,r2);
		double s1 = -1;//score(img(r1), tmp);
		double s2 = -1; //score(img(r2), tmp);
		//double s = score(img(o),tmp);
		if (r1.area()>tmp.total()) { s1 = score(img(r1), tmp); if ((s1>0 && s1>=s)) bp.push_back(rect{r1,s1}); }
		if (r2.area()>tmp.total()) { s2 = score(img(r2), tmp); if ((s2>0 && s2>=s)) bp.push_back(rect{r2,s2}); }
		//if (s1>s || s2>s) {
			bp.pop_back();
			cerr << "remove " << bp.back().r << endl;
		//}
		cerr << g << " " << bp.size() << "  " << s <<  "  " << s1 << "  " << s2 << " " << o << " "  << r1  << " " << r2 << endl;
		sort(bp.begin(), bp.end(), sort_score);

		Mat draw = img.clone();
		for (int i=0; i<bp.size(); i++) {
			rectangle(draw, bp[i].r, Scalar(200,0,0));
			cerr << g << " " << bp[i].r << " " << bp[i].s << endl;
		}
		rectangle(draw, o, Scalar(0,200,0));
		rectangle(draw, r1, Scalar(0,200,100));
		rectangle(draw, r2, Scalar(100,200,0));

		imshow("M",draw);
		imshow("t",tmp);
		if (waitKey(5000)==27) break;

	}
	for (int i=0; i<bp.size(); i++) {
		cerr << bp[i].r << " " << bp[i].s << endl;
	}
}

/*
template <class T>
struct rect {
	T t,l,b,r;
	int w() {return r-l;}
	int h() {return b-t;}
	int area() {return w()*h();}
	double s;
};

bool sort_score(const rect<int> &a, const rect<int> &b) { return a.s>b.s; }
typedef deque< rect<int> > rectmap;

double score(const Mat &m, const Mat &n) {
	return norm(mean(m),mean(n));
}

string s(const rect<int> &r) { return format("[%d %d %d %d %2.2f]",r.t,r.l,r.b,r.r,r.s); }
string S(const rect<int> &r) { return format("[%d %d %d %d %2.2f]",r.l,r.t,r.r-r.l,r.b-r.t,r.s); }
int main(int argc, char **argv) {
	Mat img = imread("dots.png");
	Mat tmp = img(Rect(31,26,25,26));
    rectmap rm(1);
    rm[0] = rect<int>{0,0,img.rows,img.cols,999999};
    for (int gen=0; gen<50; gen++) {
    	sort(rm.begin(), rm.end(), sort_score);
    	rect<int> &r = rm.front();
    	cerr << "gen " << gen << " " << rm.size() << " " << S(r) << endl;
   		rect<int> a,b;
    	if (r.h()>r.w()) { // split vertical
    		int split = r.t + r.h() / 2;
    		a.t = r.t;  a.l = r.l; a.b = split;  a.r = r.r;
    		b.t = split;  b.l = r.l; b.b = r.b;  b.r = r.r;
    	} else { // split horizontal
    		int split = r.l + r.w() / 2;
    		a.t = r.t;  a.l = r.l; a.b = r.b;  a.r = split;
    		b.t = r.t;  b.l = split; b.b = r.b;  b.r = r.r;
    	}
    	a.s = score(tmp,img(Rect(a.l,a.t,a.w(),a.h())));
    	b.s = score(tmp,img(Rect(b.l,b.t,b.w(),b.h())));
    	//cerr << "cnd a" << s(a) << endl;
    	//cerr << "cnd b" << s(b) << endl;
    	if (a.area() > 30 && a.s <= r.s) rm.push_back(a);
    	else cerr << "nil a " << S(a) << endl;
    	if (b.area() > 30 && b.s <= r.s) rm.push_back(b);
    	else cerr << "nil b " << S(b) << endl;
    	rm.pop_front();
 		for (rectmap::iterator it=rm.begin(); it!= rm.end(); it++) {
 			cerr << "   " << S(*it) << endl;
 		}
    	if (rm.size() == 1) break;
    }
    return 0;
}
*/
