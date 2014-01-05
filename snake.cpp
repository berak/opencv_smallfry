/*
   * load an image
   * draw a line of contour points around the *thing*  ( left mouse button )
   * press 'space' to run snake algo
*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include <iostream>
using namespace cv;
using namespace std;

vector<CvPoint> points;
void mouse(int k, int x, int y, int s, void* ptr) {
    if ( k == 1 ) {
        Point2f p(x,y);
        int mi = -1;
        int di=999999999;
        for ( int i=0; i<points.size(); i++ ) {
            int d = (x-points[i].x)*(x-points[i].x) + (y-points[i].y)*(y-points[i].y);
            if ( d < di ) {
                mi = 0;
                di = d;
            }
        }
        if ( points.empty() || mi==-1)
            points.push_back(p);
        else
            points.insert( points.begin() + mi, p );
    }
}

int main(int argc, char **argv) {
    int al=20,be=30,ga=400;
    namedWindow("snak",0);
    createTrackbar("alpha","snak",&al,1000);
    createTrackbar("beta","snak",&be,1000);
    createTrackbar("gamma","snak",&ga,1000);
    setMouseCallback("snak", mouse);
    Mat img = imread("t1.png");
    while(1) {
        int k = waitKey(30);
        Mat m = 255 - img; //img.clone();
        Mat mg; cvtColor(m,mg,CV_BGR2GRAY);

        if ( k == 27 ) 
            break;
        if ( k == 'c' ) 
            points.clear();
        if ( k == ' ' && (points.size() > 3) ) {
            CvTermCriteria criteria;
            criteria.type=CV_TERMCRIT_ITER; 
	        criteria.max_iter=1000; 
	        criteria.epsilon=0.1; 

            int coeffUsage = CV_VALUE;
            CvSize win = Size(3,3);
            float alpha=(float)al/1000.0f,beta=float(be)/500,gamma=float(ga)/500;
            int calcGradient = 0;
            IplImage ip = mg;

            cvSnakeImage( &ip, &(points[0]),
                      points.size(), &alpha, &beta, &gamma,
                      coeffUsage, win,
                      criteria, calcGradient );
        }
        for ( int i=1; i<points.size(); i++ ) {
            line(m,points[i-1],points[i], Scalar(0,100,0),2);
            circle(m,points[i],2,Scalar(100,0,0),1);
        }
        imshow("snak", m );
    }

    return 0;
}
