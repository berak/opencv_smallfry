#include <opencv2/opencv.hpp>
using namespace cv;

#include <iostream>
using namespace std;

//
// http://www.openu.ac.il/home/hassner/data/violentflows/violent_flows.pdf
//



Mat normalizedHist(const Mat &roi, int nbins)
{
    Mat_<float> hist(1, nbins, 0.0f);
    Mat n;
    normalize(roi, n, nbins-1, 0, NORM_MINMAX);
    float *p = n.ptr<float>();
    for (size_t i=0; i<n.total(); i++)
        hist(int(p[i])) ++;
    return hist;
}

int violentFlow(const String &filename, Mat &descriptor, int nbins=20, int frameFrom=0, int frameTo=0)
{
    VideoCapture cap(filename);
    if( !cap.isOpened() )
        return -1;

    cap.set(CAP_PROP_POS_FRAMES, frameFrom);

    Mat flow, frame;
    Mat gray, prevgray;
    Mat mag, prevmag;
    Mat accmag;
    int nFramesInMag = 0;

    for(;;)
    {
        if ((frameTo != 0) && ((int)cap.get(CAP_PROP_POS_FRAMES) == frameTo))
            break;

        cap >> frame;
        if (frame.empty())
            break;

        if (frame.rows>240)
            pyrDown(frame,frame);

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        if (! prevgray.empty())
        {
            calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            Mat xy[2];
            split(flow, xy);
            magnitude(xy[0], xy[1], mag);
        }

        if (! prevmag.empty())
        {
            Mat diffmag = mag - prevmag;

            double M;
            minMaxLoc(diffmag,0,&M,0,0);

            Mat binmag;
            threshold(diffmag, binmag, M/8, 1.0, 0);

            if (accmag.empty())
                accmag = Mat(frame.size(), CV_32F, 0.0f);

            accmag += binmag;
            nFramesInMag ++;
        }
        cerr << ".";
        std::swap(prevgray, gray);
        std::swap(prevmag, mag);
    }

    Mat hist;
    int h = accmag.rows/4;
    int w = accmag.cols/4;
    for (int r=0; r<accmag.rows; r+=h)
    {
        for (int c=0; c<accmag.cols; c+=w)
        {
            Mat roi(accmag,Rect(c,r,w,h));
            hist.push_back(normalizedHist(roi,nbins));
        }
    }
    descriptor = hist.reshape(1,1);
    return 0;
}

int main(int argc, char** argv)
{
    Mat desc1, desc2;
    violentFlow("768x576.avi",desc1,20,20,25*4);
    cerr << desc1 << endl;

    violentFlow("boxing.avi",desc2,20,20,25*4);
    cerr << desc2 << endl;

    return 0;
}
