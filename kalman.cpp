#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

void ontrack(int t,void *p)
{
    KalmanFilter * kf = (KalmanFilter*)p;
    double v = 1.0f / (t+1);
    cerr << v << endl;
    setIdentity(kf->processNoiseCov, Scalar::all(v*0.1));
    setIdentity(kf->measurementNoiseCov, Scalar::all(v)); 
    //kf->init(2,1);
}

void onmouse(int e,int x, int y, int d, void * ptr)
{
    Point3f * pt = (Point3f*)ptr;
    pt->x=float(x);
    pt->y=float(y);
    pt->z=float(d);
}

int main(int, char**)
{
    int kali=1000;
    Point3_<float> mouse;
    KalmanFilter KF(2, 2, 0);
    Mat measurement = Mat::zeros(2, 1, CV_32F);
    char code = (char)-1;
    namedWindow("Kalman",0);
    setMouseCallback("Kalman",onmouse,&mouse);
    createTrackbar("kali","Kalman",&kali,200000,ontrack,&KF);
    for(;;)
    {
        Mat img(500, 500, CV_8UC3);

        setIdentity(KF.measurementMatrix);
        setIdentity(KF.processNoiseCov, Scalar::all(.00005));
        setIdentity(KF.measurementNoiseCov, Scalar::all(0.002)); // this is inverse to the tracking speed
        setIdentity(KF.errorCovPost, Scalar::all(1));

        for(;;)
        {
            Mat prediction = KF.predict();
            if ( mouse.z == 1  )
            {
                float px = prediction.at<float>(0);
                float py = prediction.at<float>(1);
                circle( img, Point2f( px, py),  2, Scalar(20,200,20), 1, LINE_AA, 0); 
            }
            measurement.at<float>(0) = float(mouse.x);
            measurement.at<float>(1) = float(mouse.y);
            KF.correct(measurement);

            circle( img, Point2f( mouse.x, mouse.y),1, Scalar(180,180,180), 1, LINE_AA, 0); 
            imshow( "Kalman", img );

            code = (char)waitKey(10);
            if( code > 0 )
                break;
        }
        if( code == 27 || code == 'q' || code == 'Q' )
            break;
    }

    return 0;
}
