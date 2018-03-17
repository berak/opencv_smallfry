#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/dpm.hpp"

using namespace std;
using namespace cv;


int main(int argc, char**argv)
{

    cv::Ptr<dpm::DPMDetector> detector = dpm::DPMDetector::create(vector<string>(1, "inriaperson.xml"));
    CV_Assert(!detector->isEmpty());

    Mat im=imread("09.jpg", 0);

    int width = 300;
    for (int i=0; i<100; i++) {
        Rect r(i, 0, width, im.rows);
        Mat roi = im(r).clone();
   		Mat draw = im.clone();
        vector<dpm::DPMDetector::ObjectDetection> ds;
        detector->detect(roi, ds); // returns 3 channels in roi !
    	for (size_t d=0; d<ds.size(); d++) {
 	        Rect r = ds[d].rect;// r.x+=i;
 	        roi *= (0.05/(ds[d].score + 1));
 	        rectangle(roi, r, Scalar(200,0,0), 2);
 	        cerr << i << " " << ds[d].classID << " " << r << " " << r.x+i << " " << ds[d].score << endl;
    		r.x += i;
 	        rectangle(draw, r, Scalar(200,0,0), 2);
    	}
        imshow("R",roi);
        imshow("W",draw);
        if (waitKey(10)==27) return 0;
    }
    waitKey();
    return 0;
}
