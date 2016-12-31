#include <opencv2/opencv.hpp>
#include <iostream>
#include <deque>
#include <map>
using namespace cv;
using namespace std;

/***

eye-blink classification, based on:
http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

using data from:
http://www.icg.tugraz.at/Members/divjak/prework/PreWork-Data
http://www.icg.tugraz.at/Members/divjak/prework/groundtruth_talking.zip
http://www-prima.inrialpes.fr/FGnet/data/01-TalkingFace/talking_face.html
(annotated landmarks + ground truth (blink/non-blink) data for the time series)

note, that the landmark model used here differs from the paper,
as it only has 1 vertical point pair, so the ear is calulated as
cv::norm(p27 - p29) / cv::norm(p28 - p30); (for the left eye)

accuracy: 0.963904
confusion:
[2638, 72; // neg
 36, 246]  // pos

***/

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
        }
    }

    float auc(vector<Point2f> &roc) {
        float _auc = 0.0f;
        for (int i=0; i<int(roc.size())-1; i++) {
            _auc += (roc[i+1].y + roc[i].y) * (roc[i+1].x - roc[i].x);
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

Mat sigmoid(const Mat &m) {
    Mat P;
    multiply(m, -1, P);
    exp(P, P);
    add(P, 1, P);
    divide(1, P, P);
    return P;
}

//
// parse precalculated landmarks & ground truth data from
//    http://www.icg.tugraz.at/Members/divjak/prework/groundtruth_talking.zip
//    http://www-prima.inrialpes.fr/FGnet/data/01-TalkingFace/talking_face.html
//    (5000 frames with 63 blinks)
//    and write a csv file compatible with opencv's traindata
//
void writeEAR() {
    string blink = "C:/data/blink/";
    string dummy;

    // there are 5000 frames in the db, first assume
    // they're all "non-blink":
    vector<int> tlabels(5000,0);
    // then fill in the ground-truth positives:
    ifstream anno(blink + "TALKING_groundtruth_PREWORK.txt");
    getline(anno, dummy);
    for (int i=0; i<63; i++) {
        int a,b,c;
        anno >> dummy;
        anno >> a >> b >> c;
        if (dummy == "F:") { // only "Full blink used here."
            fill(tlabels.begin() + a, tlabels.begin() + c, 1);
        }
    }

    // parse 5000 annotation(landmarks) files,
    // select p27,p28,p29,p30 for left eye,
    // and    p32,p33,p34,p35 for right eye.
    // calculate eye-aspect-ratio (ear) for both,
    // and write a 13-element time-series to csv.
    ofstream csv("ear.csv");
    deque<float> el,er;
    for (size_t i=0; i<5000; i++) {
        ifstream in(blink + format("points/franck_%05d.pts", i));
        if (! in.good()) {
            cerr << "bad file " << i << endl;
            continue;
        }
        // skip annotation header and first 27 points (30 lines)
        string dummy;
        for (int j=0; j<30; j++)
            getline(in, dummy);

        // they start counting at 1, so the 1st eye corner is point 27
        Point2d p27,p28,p29,p30;
        in >> p27.x >> p27.y;
        in >> p28.x >> p28.y;
        in >> p29.x >> p29.y;
        in >> p30.x >> p30.y;
        double ear_l = norm(p27 - p29) / norm(p28 - p30);
        el.push_back(ear_l);

        // skip point 31
        getline(in, dummy);

        Point2d p32,p33,p34,p35;
        in >> p32.x >> p32.y;
        in >> p33.x >> p33.y;
        in >> p34.x >> p34.y;
        in >> p35.x >> p35.y;
        double ear_r = norm(p34 - p32) / norm(p33 - p35);
        er.push_back(ear_r);

        // write out queues for left & right eye
        // "backdated" to t-6 ! (center of the 13 elem queue)
        if (i >= 13) {
            csv << tlabels[i-6] << ",";
            for (int k=0; k<13; k++)
                csv << el[k] << ",";
            csv << endl;

            csv << tlabels[i-6] << ",";
            for (int k=0; k<13; k++)
                csv << er[k] << ",";
            csv << endl;

            el.pop_front();
            er.pop_front();
        }
    }
}


int main(int argc, char **argv)
{
    ifstream e("ear.csv");
    if (! e.good()) {
        writeEAR();
    }

    // train & test with SVM:
    Ptr<ml::SVM> svm = ml::SVM::create();
    Ptr<ml::TrainData> tdata = ml::TrainData::loadFromCSV("ear.csv",0,0,1);
    tdata->setTrainTestSplitRatio(0.7);

    Mat data = tdata->getTrainSamples();
    Mat labels = tdata->getTrainResponses();
    labels.convertTo(labels, CV_32S); // hrrm!
    svm->train(data,0,labels);

    Mat vdata = tdata->getTestSamples();
    Mat gdtruth = tdata->getTestResponses();
    Mat predict;
    svm->predict(vdata, predict);

    float correct = countNonZero(predict == gdtruth);
    float accuracy = correct / predict.total();
    cerr << "accuracy: " << accuracy << endl;

    // accuracy alone is not enough here, since it might
    // simply have missed all positives !
    Mat_<int> confusion(2,2,0);
    for (int i=0; i<predict.rows; i++) {
        int p = (int)predict.at<float>(i);
        int t = (int)gdtruth.at<float>(i);
        confusion(p,t) ++;
    }
    cerr << "confusion:\n" << confusion << endl;

    // additionally, do ROC analysis.
    // we need raw output, so another prediction required:
    svm->predict(vdata, predict, ml::StatModel::RAW_OUTPUT);
    // svm gives distances, needed are probs in [0..1]
    predict = sigmoid(-predict); // positive features have negative distance

    std::vector<Point2f> roc;
    ROC::curve(predict, gdtruth, roc, 100);
    cerr << "AUC " << ROC::auc(roc) << endl;

    Mat roc_draw(480, 640, CV_8UC3, Scalar::all(255));
    ROC::draw(roc, roc_draw, Scalar(255,0,0));
    imshow("ROC", roc_draw);
    waitKey();

    return 0;
}
