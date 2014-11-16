#include "opencv2/opencv.hpp"
#include "opencv2/latentsvm.hpp"
using namespace cv;

#include <iostream>
using namespace std;


int main()
{
    int64 t0 = getTickCount();
    vector<string> fn(1,"person.xml"); 
    // note, that you can no more use the xml files from opencv_extra, 
    // you will have to use the ones supplied in opencv_contrib/modules/latensvm/testdata 
    Ptr<lsvm::LSVMDetector> late = lsvm::LSVMDetector::create(fn);
    if ( late->isEmpty() ) 
    {
        cerr << "something wrong with the xml ?" << endl;
        return 1;
    }
    int64 t1 = getTickCount();
    Mat m = imread("stylish-people-sitting-chatting-chattingpeople-white-background-31667283.jpg");
    vector<lsvm::LSVMDetector::ObjectDetection> dets;
    late->detect(m,dets,0.2f);
    int64 t2 = getTickCount();
    cerr << dets.size() << " detections, " << ((t1-t0)/getTickFrequency()) << " / " << ((t2-t1)/getTickFrequency()) << " seconds." <<endl;
    for (size_t i=0; i<dets.size(); ++i)
    {
        cerr << dets[i].classID << " " << dets[i].rect << " " << dets[i].score << endl;
        if (dets[i].score>0)
            rectangle(m,dets[i].rect,Scalar(0,200,0));
        else if (dets[i].score>-1.0)
            rectangle(m,dets[i].rect,Scalar(0,0,200));
    }
    imshow("m",m);
    waitKey();
    return 0;
}
