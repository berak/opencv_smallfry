#include "opencv2/opencv.hpp"
#include <deque>

using namespace std;
using namespace cv;

namespace freeman {

// transform 8 possible dirs to 6, then 3 required bits, keeping it clockwise ordered
uchar encode(const Point &a, const Point &b) {
    uchar up    = (a.y > b.y);
    uchar left  = (a.x > b.x);
    uchar down  = (a.y < b.y);
    uchar right = (a.x < b.x);
    uchar equx  = (a.y == b.y);
    uchar equy  = (a.x == b.x);

    return (up    && equy)  ? 0 : // N
           (up    && right) ? 1 : // NE
           (right && equx)  ? 2 : // E
           (down  && right) ? 3 : // SE
           (down  && equy)  ? 4 : // S
           (left  && down)  ? 5 : // SW
           (left  && equx)  ? 6 : // W
                              7 ; // NW
}

// forward pass
void chain(const vector<Point> &contours, vector<uchar> &_chain) {
    int i=0;
    for (; i<contours.size()-1; i++) {
        _chain.push_back(encode(contours[i],contours[i+1]));
    }
    _chain.push_back(encode(contours[i],contours[0]));
}
Mat chain_image(const vector<vector<Point> > &contours, const Size &size) {
    Mat img(size, CV_8U, Scalar(0));
    for (int c=0; c<contours.size(); c++) {
        int i=0;
        for(; i<contours[c].size()-1; i++) {
            img.at<uchar>(contours[c][i]) = encode(contours[c][i], contours[c][i+1]);
        }
        img.at<uchar>(contours[c][i]) = encode(contours[c][i], contours[c][0]);
    }
    return img;
}

// backward pass
Point next(const Point &p, uchar code) {
    int o[8*2] = {-1,0, -1,1, 0,1, 1,1, 1,0, 1,-1, 0,-1, -1,-1};
    return Point(p.x+o[code*2], p.x+o[code*2+1]);
}
void reconstruct(const vector<uchar> &_chain, vector<Point> &contours, Point offset, float scale=1.0) {
    for (int i=0; i<_chain.size(); i++) {
        Point p = next(offset, _chain[i] * scale);
        contours.push_back(p);
        offset = p;
    }
}

} // freeman

int main ()
{
    Mat img = imread("img/shadow.png",0);
    pyrUp(img, img);
    Mat bin = img<30;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(bin, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0));
    Mat draw;
    cvtColor(img,draw,COLOR_GRAY2BGR);
    drawContours(draw, contours, -1, Scalar(0, 255, 0), 1);
    for (int i=0; i<contours.size(); i++) {
        vector<Point> &c = contours[i];
        if (contourArea(c)>50) {
            for (int j=0; j<c.size(); j++)
                circle(draw,c[j],1,Scalar(0,0,255),1);
            vector<uchar> v;
            //freeman::chain(c,v);
            //cerr << Mat(v).t() << endl;
        }
    }
    imshow("A",freeman::chain_image(contours, img.size())*32);
    imshow("W",draw);
    waitKey();
    return 0;
}
