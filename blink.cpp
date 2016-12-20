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


int main(int argc, char **argv)
{
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
        // skip first 30 lines (3 lines offset + 27)
        string dummy;
        for (int j=0; j<30; j++)
            getline(in, dummy);

        Point2d p27,p28,p29,p30;
        in >> p27.x >> p27.y;
        in >> p28.x >> p28.y;
        in >> p29.x >> p29.y;
        in >> p30.x >> p30.y;
        double ear_l = cv::norm(p27 - p29) / cv::norm(p28 - p30);
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

    // train & test with SVM:
    Ptr<ml::SVM> svm = ml::SVM::create();
    Ptr<ml::TrainData> tdata = ml::TrainData::loadFromCSV("ear.csv",0,0,1);
    tdata->setTrainTestSplitRatio(0.7);

    Mat data = tdata->getTrainSamples();
    Mat labels = tdata->getTrainResponses();
    labels.convertTo(labels, CV_32S); // hrrm!
    svm->train(data,0,labels);

    Mat vdata = tdata->getTestSamples();
    Mat vlabels = tdata->getTestResponses();
    Mat results;
    svm->predict(vdata,results);

    float correct = countNonZero(results == vlabels);
    float accuracy = correct / results.total();
    cerr << "accuracy: " << accuracy << endl;

    // accuracy alone is not enough here, since it might
    // simply have missed all positives !
    Mat_<float> confusion(2,2);
    for (int i=0; i<results.rows; i++) {
        int p = (int)results.at<float>(i);
        int t = (int)vlabels.at<float>(i);
        confusion(p,t) ++;
    }
    cerr << "confusion:\n" << confusion << endl;
    return 0;
}