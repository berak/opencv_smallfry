//#include "WinBase.h"

#include "dbrief.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <time.h>
#include <vector>
#include <cmath>
#include <iostream>
using namespace cv;
using namespace std;

//
// hi @Gil, the hidden question is  here:
// what is the way to do this properly ? how do we match ?
// ideally, i'd want some knn match over something like a combined
//        hamming distance for the features 
//        L2 keypoint distance. 
//    (think of superpixels)
// really , below code is total bogus
//
double match( const vector< bitset<32> > & desc1, const vector< bitset<32> > & desc2 )
{
    Mat_<int> mhist=Mat_<int>::zeros(1,8); // no idea, just seems useful for vizualisation
    size_t nmax = min(desc1.size(),desc2.size());
    double d = 0;
    for ( size_t i=0; i<nmax; i++ )
    {
        int hd = CVLAB::HAMMING_DISTANCE(desc1[i],desc2[i]);;
        mhist(hd) += 1;
        d += double(hd);
    }
    cerr << d << " " << mhist << endl;;
    return d/nmax;
}

// Detect keypoints of img with FAST and store them to kpts given the threshold kptDetectorThreshold.
int extractKeypoints(std::vector< cv::KeyPoint >& kpts, int kptDetectorThreshold, const cv::Mat & img)
{
    cv::Rect r = cv::Rect(CVLAB::IMAGE_PADDING_LEFT, CVLAB::IMAGE_PADDING_TOP,
		    CVLAB::SUBIMAGE_WIDTH(img.cols), CVLAB::SUBIMAGE_HEIGHT(img.rows));

    // Use FAST corner detector to detect the image keypoints
    cv::Mat roi = img(r);//.clone();
    cv::FAST( roi, kpts, kptDetectorThreshold, true);

    // Transform the points to their actual image coordinates:
    for (unsigned int i = 0, sz = kpts.size(); i < sz; ++i)
        kpts[i].pt.x += CVLAB::IMAGE_PADDING_LEFT, kpts[i].pt.y += CVLAB::IMAGE_PADDING_TOP;

    return kpts.size();
}

void draw(cv::Mat & m, const vector<cv::KeyPoint> & kpt1, const vector<cv::KeyPoint> & kpt2 )
{
    size_t nmax = min(kpt1.size(),kpt2.size());
    for (size_t i=0; i<nmax; ++i)
    {
        cv::Point p1(kpt1[i].pt), p2(kpt2[i].pt);
        if ( abs(p1.x-p2.x)+abs(p1.y-p2.y) < 100 )
            line(m,p1,p2,Scalar(30,120,20));
        //else
        //    line(m,p1,p2,Scalar(30,30,70));
        circle(m,p1,2,Scalar(120,20,20));
        circle(m,p2,2,Scalar(30,20,120));
    }
}

void proc(CVLAB::Dbrief & db, cv::Mat & m, int thresh, vector<cv::KeyPoint> & kpt, vector< bitset<32> > & desc )
{
    kpt.clear();
    extractKeypoints(kpt, thresh, m);

    desc.clear();
    db.getDbriefDescriptors(desc,kpt,m);
    cerr << "kpts: " << kpt.size() << "\t desc: " << desc.size() << endl;
}

int main()
{
    vector< bitset<32> > desc1,desc2;
    vector<cv::KeyPoint> kpt1, kpt2;

    CVLAB::Dbrief db;

    if ( 1 )
    {
        Mat m1 = imread("../demo/left01.jpg",0); // infamous selfies with checker board 
        Mat m2 = imread("../demo/right01.jpg",0);
        cerr << m1.rows<< " " << m1.cols << endl;

        proc(db,m1,13,kpt1,desc1);
        proc(db,m2,13,kpt2,desc2);
        cerr << match(desc1,desc2)<< endl;
        draw(m1,kpt1,kpt2);
        imshow("dbrief",m1);
        waitKey();
    }
    else 
    {
        VideoCapture cap(0);
        Mat gray,bgr,bgr2;
        while ( cap.isOpened() )
        {
            Mat bgr ;
            cap >> bgr;
            cvtColor(bgr,gray,CV_BGR2GRAY);

            proc( db, gray, 9, kpt1, desc1 );

            if ( desc2.size() )
            {
                cerr << match(desc1,desc2)<< endl;
                bgr2 = bgr.clone();
                draw(bgr,kpt1,kpt2);
                imshow("dbrief",bgr);
            }
            desc2=desc1;
            kpt2=kpt1;

            int k = cv::waitKey(5);
            if ( k == 27 || k == 'q' ) 
                break;
        }
    }
    return 0;
}