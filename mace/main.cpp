#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include "mace.h"

using namespace cv;
using namespace std;

namespace ROC {
    void curve(const Mat &probs, const Mat &truth, vector<Point2f> &roc, int N, const float eps=1e-1) {
        for (int i=0; i<N; i++) {
            float thresh = float(N-i) / N;
            float TP = countNonZero((probs >  thresh) & (truth >  eps));
            float TN = countNonZero((probs <= thresh) & (truth <= eps));
            float FP = countNonZero((probs >  thresh) & (truth <= eps));
            float FN = countNonZero((probs <= thresh) & (truth >  eps));
            float FPR = FP / (FP + TN);
            float TPR = TP / (TP + FN);
            roc.push_back(Point2f(FPR, TPR));

            //// use precision / recall, than FPR / TPR
            //float recall    = TP / (TP + FN);
            //float precision = TP / (TP + FP);
            //roc.push_back(Point2f(recall, precision));
        }
    }

    float auc(vector<Point2f> &roc) {
        float _auc = 0.0f;
        for (int i=0; i<int(roc.size())-1; i++) {
            _auc += (roc[i+1].y + roc[i].y) * (roc[i+1].x - roc[i].x); // riemann
        }
        return _auc * 0.5f;
    }

    void draw(vector<Point2f> &roc, Mat &img, const Scalar &color) {
        int   N = roc.size();
        float SY = float(img.rows) / N;
        float SX = float(img.cols) / N;
        Point2f prev;
        for (size_t i=0; i<roc.size(); i++) {
            Point2f cur(roc[i].x*N*SX, (1.0-roc[i].y)*N*SY); // opencv y axis points down
            if (i>0)
                line(img, prev, cur, color, 1);
            prev = cur;
        }
    }
} // ROC

extern bool DBGDRAW;
//
// all-against-all test on att faces
//
int main(int argc, char **argv) {
    CommandLineParser parser(argc, argv,
            "{ help h usage ? ||     show this message }"
            "{ thresh t       |0.42| threshold }"
            "{ subj s         |40|   num subjects in test(max: 40) }"
            "{ ocl o          ||     use opencl }"
            "{ siz z          |64|   image size }"
            "{ debug d        ||     show debug images }"
            "{ random r       ||     random convolute }"
    );
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    bool useocl = parser.has("ocl");
    ocl::setUseOpenCL(useocl);
    int N = parser.get<int>("subj");
    int Z = parser.get<int>("siz");
    float thresh = parser.get<float>("thresh");
    bool randomize = parser.has("random");
    DBGDRAW = parser.has("debug");

    String att = "c:/data/faces/att/";
    cerr << att << " " << N << " " << thresh << " " << randomize << " " << useocl << endl;
    Mat_<int> confusion(N,N,0);
    float tp=0,fp=0,tn=0,fn=0;
    Mat predict, truth;
    for (int p=1; p<=N; p++) { // att starts counting from one ..
        vector<Mat> p1;
        for (int i=1; i<=10; i++) {
            Mat i1 = imread(att + format("s%d/%d.pgm", p, i), 0);
            p1.push_back(i1);
        }
        Ptr<MACE> mace = MACE::create(Z, (randomize ? p+123 : 0));
        mace->compute(p1);
        for (int q=1; q<=N; q++) {
            for (int i=1; i<=10; i++) {
                Mat i2 = imread(att + format("s%d/%d.pgm", q, i), 0);
                double pred = mace->correlate(i2);
                predict.push_back(pred);
                truth.push_back(p==q);
                if (pred > thresh) {
                    confusion(p-1,q-1) ++; // positively detected.
                    if (p==q)  tp ++;
                    else       fp ++;
                } else {
                    if (p==q)  fn ++;
                    else       tn ++;
                }
            }
        }
        cerr << ".";
    }
    if (N<=16)
        cerr << endl << confusion << endl;
    else
        cerr << endl << confusion(Rect(0,0,16,16)) << endl;
    float acc = (tp+tn) / (tp+tn+fp+fn);
    cerr << "tp: " << tp << " fp: " << fp << " tn: " << tn << " fn: " << fn << " acc: " << acc << endl;
    vector<Point2f> roc;
    ROC::curve(predict,truth,roc,400);
    float auc = ROC::auc(roc);
    cerr << "auc: " << auc << endl;
    if (DBGDRAW) {
        Mat img(400, 400, CV_8UC3, Scalar::all(127));
        ROC::draw(roc, img, Scalar(0,0,150));
        imshow("ROC",img);
        waitKey();
    }
    return 1;
}
