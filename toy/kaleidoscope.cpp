//https://github.com/SaranshKejriwal/Kaleidoscope

#include<opencv2/opencv.hpp>
using namespace cv;

#include<iostream>
using namespace std;


void kaleidoscope(const Mat &img, Mat &result)
{
    Mat img2,kal1(400,400,CV_8UC3,Scalar(0,0,0));
    Mat kal2,add;
    resize(img,img2,Size(400,400));
    for(int i=0,j=0; i<200; i++)
    {
        for(j=0; j<i; j++)
        {
            kal1.at<Vec3b>(Point(i,j))=img2.at<Vec3b>(Point(i,j)); //gives 1/8 of img
        }
        kal1.at<Vec3b>(Point(i,j))= 0.5 * img2.at<Vec3b>(Point(i,j)); //interpolate overlapping X
    }
    flip(kal1,kal2,0);
    addWeighted(kal1,1,kal2,1,0,add);
    flip(kal2,kal2,1);
    addWeighted(add,1,kal2,1,0,add);
    flip(kal2,kal2,0);
    addWeighted(add,1,kal2,1,0,add);
    transpose(kal2,kal2);
    addWeighted(add,1,kal2,1,0,add);
    flip(kal2,kal2,0);
    addWeighted(add,1,kal2,1,0,add);
    flip(kal2,kal2,1);
    addWeighted(add,1,kal2,1,0,add);
    flip(kal2,kal2,0);
    addWeighted(add,1,kal2,1,0,add);
    result = add;
}


int main()
{
    int f=-1, t=0, k=1;

    VideoCapture cap(0);
    if (! cap.isOpened())
        return -1;

    Mat frame;
    while(cap.read(frame))
    {
        if (f>-1) flip(frame, frame, f);
        if (t) frame = frame.t();
        if (k) kaleidoscope(frame, frame);
        imshow("kaleidoscope", frame);
        int c = waitKey(30);
        if (c == 27 ) break;
        if (c == 'f') { if (++f > 1) f=-1; }
        if (c == 'g') { t = ! t; }
        if (c == 'k') { k = ! k; }
    }
    return 1;
}
