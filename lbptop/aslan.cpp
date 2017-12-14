#include <iostream>
#include "opencv2/opencv.hpp"
#include "lbptop.h"

using namespace cv;
using namespace std;

Mat dist(const Mat &a, const Mat &b) {
    Scalar m,d;
    meanStdDev(a, m, d);
    Mat a2; divide((a - m), d, a2);
    meanStdDev(b, m, d);
    Mat b2; divide((b - m), d, b2);
    Mat c = a2 - b2;
    multiply(c,c,c);
    return c; // (a-b)^2
}

int main() {
    Mat data;
    if (1) {
        String vids = "c:/data/video/ASLAN_AVI/*.avi";
        vector<String> fn;
        glob(vids, fn);
        for (size_t i=0; i<fn.size(); i++) {
            Mat desc;
            if (lbpFlow(fn[i], desc, 0, 40) < 0)
                continue;
            data.push_back(desc);
            cerr << fn[i] << '\r';
        }
        FileStorage fs("aslan_lbp.yml.gz", 1);
        fs << "feat" << data;
    }
    cerr << "\n reading data." << endl;
    FileStorage fs("aslan_lbp.yml.gz", 0);
    fs["feat"] >> data;
    fs.release();
    cerr << data.size() << " features." << endl;
    float acc = 0.0f;
    for (int split=0; split<10; split++) {
        Mat train_s, train_l;
        Mat test_s,  test_l;
        ifstream tp("view2.txt");
        int k=0, i1, i2, same, dummy;
        while(tp >> i1 >> i2 >> same >> dummy >> dummy) {
            if ((k >= split*600) && (k < (split+1)*600)) {
                test_s.push_back(dist(data.row(i1-1), data.row(i2-1)));
                test_l.push_back(same);
            } else {
                train_s.push_back(dist(data.row(i1-1), data.row(i2-1)));
                train_l.push_back(same);
            }
            k++;
        }

        Ptr<ml::SVM> svm = ml::SVM::create();
        svm->setKernel(ml::SVM::LINEAR);
        svm->train(train_s, 0, train_l);

        Mat r;
        svm->predict(test_s, r);
        r.convertTo(r, CV_32S);
        float a = (float)countNonZero(r==test_l)/test_s.rows;
        cerr << "split " << split << "  " << a << endl;
        acc += a;
    }
    cerr << "final    " << acc*0.1f << endl;
    return 0;
}

