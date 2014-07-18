#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
using namespace cv;
using namespace std;


int main()
{
    vector<String> fn(1,"person.xml"); // from itseez/opencv_data repo
    LatentSvmDetector late;
    bool ok = late.load(fn);
    if ( ! ok ) 
    {
        return 1;
    }
    //VideoCapture cap("E:/code/opencv/samples/gpu/768x576.avi");
    VideoCapture cap("D:/video/keep/Stranger.Than.Paradise.1984.Jim.Jarmusch.avi");
    int frameid=31310;
    cap.set(cv::CAP_PROP_POS_FRAMES, frameid );
    while(cap.isOpened())
    {
        Mat m; cap >> m;
        if (m.empty()) 
            break;
        //resize(m,m,Size(m.cols*2/3,m.rows*2/3)); // does not detect
        //Mat g; cvtColor(m,g,COLOR_BGR2GRAY);     // throws
        vector<LatentSvmDetector::ObjectDetection> dets;
        late.detect(m,dets,0.2f);
        cerr << "frame " << frameid << ",  " << dets.size() << " detections." << endl;
        for (size_t i=0; i<dets.size(); ++i)
        {
            cerr << dets[i].classID << " " << dets[i].rect << " " << dets[i].score << endl;
            if (dets[i].score>0)
                rectangle(m,dets[i].rect,Scalar(0,200,0));
            else if (dets[i].score>-1.0)
                rectangle(m,dets[i].rect,Scalar(0,0,200));
        }
        imshow("m",m);
        if ( waitKey(6) == 27 )
            break;;
        frameid = cap.get(cv::CAP_PROP_POS_FRAMES);
        cap.set(cv::CAP_PROP_POS_FRAMES,frameid+947);

    }
   return 0;
}
