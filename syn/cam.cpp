#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;


struct msg {
	int c;
	int e;
	float v;
	msg(int c, int e, float v) : c(c),e(e),v(v) { /*cerr << str() << endl;*/}
	string str() { return format("%d %d %2.4f",c,e,v); }
};

int main()
{
	VideoCapture cap(0);
	int ticks=0;
	Mat frame,mask,hsv;
	while(1) {
		cap >> frame;
		if (frame.empty())
			break;
		cvtColor(frame,hsv,COLOR_BGR2HSV);
		//Scalar up(80,70,200), lo(60,30,150);
		Scalar up(98,90,180), lo(78,80,150);
		inRange(hsv,lo,up,mask);
		imshow("M",mask);

	    vector<vector<Point>> contours;
	    vector<Vec4i> hierarchy;
		blur(mask,mask,Size(9,9));
	    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
	    Rect r;
	    double A=0; int id=-1;
	    for (int i=0; i<contours.size(); i++) {
	        vector<Point> &c = contours[i];
	        double a = contourArea(c);
	        if (a>A) {
	        	A = a;
	            id = i;
	        	r = boundingRect(c);
	        }
	    }

	    drawContours(frame, contours, id, Scalar(0, 155, 0), 1);
	    line(frame,Point(frame.cols/2,0),Point(frame.cols/2,frame.rows-1),Scalar(150,150,0),3);
	    line(frame,Point(0,frame.rows/2),Point(frame.cols-1,frame.rows/2),Scalar(150,150,0),3);
	    Point c(r.x+r.width/2, r.y+r.height/2);
	    if (c.x>0) {
			//cerr << c << endl;
			int W = (frame.cols/2);
			int H = (frame.rows/2);
		    // 0 | 1
		    // 2 | 3
		    if (c.x < W) {
			    if (c.y < H) {
					cout << msg(0,2,float(c.x)/W).str() << endl;
					cout << msg(0,1,float(H-c.y)/H).str() << endl;
			    } else {
					cout << msg(2,2,float(c.x)/W).str() << endl;
					cout << msg(2,1,float(2*H-c.y)/H).str() << endl;
			    }
		    } else {
			    if (c.y < frame.rows/2) {
					cout << msg(1,2,float(c.x-W)/W).str() << endl;
					cout << msg(1,1,float(H-c.y)/H).str() << endl;
			    } else {
					cout << msg(3,2,float(c.x-W)/W).str() << endl;
					cout << msg(3,1,float(2*H-c.y)/H).str() << endl;
			    }
		    }
		}
	    circle(frame,c,5,Scalar(0,0,200),-1);
		imshow("F",frame);
		if (waitKey(10)>0) break;
	}
	cout << "9 9 9" << endl;
	return 0;
}
