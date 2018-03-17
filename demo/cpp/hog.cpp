
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
using namespace std;
using namespace cv;
#include "profile.h"


int main(int argc, char** argv)
{
    ocl::setUseOpenCL(true);
    cout << "ocl " << cv::ocl::useOpenCL() << endl;
	String iname=argc<2?"C:\\data\\img\\cache\\12\\3E38FDF3C5602FFDECD09242404C7A2D34249659.png":argv[1];
    Mat img = imread(iname, 0);

    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    std::vector<Rect> rects;
    std::vector<double> weights;
    //hog.detectMultiScale(img,rects,weights);
    hog.detectMultiScale(img, rects, 0, Size(8,8), Size(32,32), 1.05, 2);
    cerr <<  rects.size() <<  " detected. " << endl;

    cvtColor(img,img,COLOR_GRAY2BGR);
    for (size_t i=0; i<rects.size(); i++) {
      //  if (rects.size() < 200) cerr << weights[i] << " " << rects[i] << endl;
        rectangle(img,rects[i],Scalar(theRNG().uniform(0,255),theRNG().uniform(0,255),theRNG().uniform(0,255)),2);
    }
    imshow("I",img);
    waitKey();
    return 0;
}

