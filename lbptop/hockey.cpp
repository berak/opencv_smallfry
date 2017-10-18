#include <iostream>
#include "opencv2/opencv.hpp"
#include "lbptop.h"

using namespace cv;
using namespace std;

// ----------8<---------------------------------------------------------
// http://www.openu.ac.il/home/hassner/data/violentflows/violent_flows.pdf

// bs 24: 84 87
// bs 30: 85 87
// bs 40: 91 90
// 4x4 blocks in xy: 92 85
int main() {
    Mat data;
    if (1) {
        String vids = "c:/data/video/HockeyFights/*.avi"; // 500 fi, 500 no
        vector<String> fn;
        glob(vids, fn);

        for (size_t i=0; i<fn.size(); i++) {
            Mat desc;
            if (-1 == lbpFlow(fn[i], desc,0,40))
                continue;
            data.push_back(desc);
            cerr << fn[i] <<  " " << desc.size() << endl;
        }
        cerr << "writing " << data.size() << " elems, " << data.total() * data.elemSize() << " bytes." << endl;
        FileStorage fs("hockey_lbpflow.yml.gz", 1);
        fs << "lbp" << data;
        fs.release();
    }

    FileStorage fs("hockey_lbpflow.yml.gz", 0);
    fs["lbp"] >> data;
    int NT = 100; // num tests
    int COLS = data.cols;
    Mat labels(1000-2*NT,1,CV_32S, Scalar(1)); // keep first & last 50 for testing
    labels(Rect(0,500-NT,1,500-NT)) = 0;

    cerr << "start training." << endl;
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setKernel(ml::SVM::LINEAR);
    svm->train(data(Rect(0,NT,COLS,1000-2*NT)), 0, labels);
    cerr << "start predict." << endl;
    Mat rpos,rneg;
    svm->predict(data(Rect(0,0,      COLS,NT)), rpos);
    svm->predict(data(Rect(0,1000-NT,COLS,NT)), rneg);
    cerr << countNonZero(rpos==1) << endl;
    cerr << countNonZero(rneg==0) << endl;

    return 0;
}


