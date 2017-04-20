#include <opencv2/opencv.hpp>
//#include <opencv2/plot.hpp>
#include <iostream>
using namespace cv;
using namespace std;


Vec2d spline(double u, const vector<double> &x, const vector<double> &y, bool m)
{
    double B[4];
    int np = x.size();
    int ns = np / 4;
    int s  = np - 4;
    if ( u < 1.0f )
        s = (int)( u * (np-1) );

    if ( m ) {
        // bspline
        double u1  = ( u - (double)s / (np-1) ) * ns;
        double u2  = 1.0f - u1;
        double u11 = u1*u1;
        double u22 = u2*u2;
        B[0] = u22  * u2;
        B[1] = 3.0f * u1  * u22;
        B[2] = 3.0f * u11 * u2;
        B[3] = u11  * u1;
    } else {
        // catmullRom
        double t1 = (1-u)*(1-u);
        double t2 = u*u;
        double t3 = u*u*u;
        B[0] = (-u*t1)*0.5f;
        B[1] = ((2-5*t2+3*t3))*0.5f;
        B[2] = (u*(1+4*u-3*t2))*0.5f;
        B[3] = (t2*(1-u))*0.5f;
        s -= 1;
    }

    Vec2d res(0,0);
    for ( int a=0; a<4; a++ ) {
        res[0] += x[s] * B[a];
        res[1] += y[s] * B[a];
        s++;
    }
    return res;
}


int main(int argc, char **argv)
{
    int m = argc>1&&argv[1][0]=='c' ? 0 : 1;
    cerr << "mode " << m ;
    // generate data
    vector<double> x{0.03,0.1,0.4,0.7,0.7,0.7,1,1};
    vector<double> y{.0,.62,.5,1.,1,1,1,1};
    Mat plotMat(400,400,CV_8UC3,Scalar::all(0));
    int N=100;
    Point po = Point(0,400);
    for (int i=0; i<x.size(); i++) {
        Point p(x[i]*200, y[i]*200);
        p.y = 400 - p.y;
        line(plotMat,po,p,Scalar(0,200,0),1);
        po = p;
    }
    Point off(5,5);
    po = Point(0,400);
    for (int i=0; i<N; i++) {
        double u = double(i)/N;
        Vec2d p = spline(u, x, y, m);
        Point pt(p*200);
        pt += off;
        pt.y = 400 - pt.y;
        pt.x;
        line(plotMat,po,pt,Scalar(200,0,0),1);
        po = pt;
        cerr << p <<  " " << u << endl;
    }


    imshow("MTF", plotMat);
    waitKey();
/*
    // plot
    Mat plotMat;
    Ptr<plot::Plot2d> plot;
    plot = plot::createPlot2d(x, y);
    plot->setMaxX(2);
    plot->setMinX(-1);
    plot->setMaxY(2.);
    plot->setMinY(-2.);
    plot->render(plotMat);
    imshow("MTF", plotMat);
    imwrite("./plot.jpg", plotMat);
    waitKey();
*/
    return 0;
}
