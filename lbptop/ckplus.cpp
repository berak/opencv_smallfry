#include "opencv2/opencv.hpp"
#include <iostream>
#include "lbptop.h"

using namespace cv;
using namespace std;


int make_data(bool singleImage=false, bool useDiff=false) {
    CascadeClassifier cad("haarcascade_frontalface_alt.xml");
    String path = "c:/data/faces/ckplus/";
    vector<String> emos; // only 327 of all sequences are labelled, we have to check those 1st.
    glob(path+"Emotion/*.txt", emos, true);
    cerr << emos.size() << " " << singleImage << " " << useDiff << endl;
    vector<float> histo(8,0);
    Mat data,labels;
    for (auto f:emos) {
        ifstream emo(f);
        float e;
        emo >> e;
        histo[(int)e] ++;
        int l = f.find_last_of('\\');
        String txt = f.substr(l + 1);
        String sub = txt.substr(0, 4);
        String em  = txt.substr(5, 3);
        String frm = txt.substr(9, 8);
        cerr << e << " " << sub << " " << em << " " << frm << '\r';
        String imgbase = path + "cohn-kanade-images/" + sub + "/" + em + "/";
        int nfrm = stoi(frm);
        Sequence seq;
        Rect box;
        Mat first, dsum;
        for (int f=0; f<nfrm; f++) {
            Mat gray;
            String iname = imgbase + sub + "_" + em + "_" + format("%08d",f+1) +".png";
            gray = imread(iname, 0);
            if (gray.empty())
                continue;
            equalizeHist(gray,gray);
            if (f==0) {
                vector<Rect> faces;
                cad.detectMultiScale(gray,faces,1.1,4,CV_HAAR_FIND_BIGGEST_OBJECT,cv::Size(30,30));
                if (faces.size()) {
                    box = faces[0];
                    box.height *= 1.05; // cascade tends to cut off mouth
                }
            }
            Mat det;
            resize(gray(box), det, Size(200,200));
            if (useDiff && f==0) {
            	first = det.clone();
                dsum = Mat::zeros(det.size(), CV_64F);
                cerr << "*" << endl;
            	continue;
            }
            if (useDiff) {
                blur(det,det,Size(4,4));
                Mat d1;
                absdiff(det,first,d1);
                //threshold(det,det,20,255,0);
                //Mat d1 = det - first;
                first = det.clone();
                det = d1-2;
                cv::accumulate(d1,dsum);
//  cerr << f << " " << sum(d1)[0] << " " << sum(det)[0] << " " << sum(first==det)[0] << " " << sum(dsum)[0] << endl;
            }
            if (singleImage) {
                int NB=4;
                int w = det.cols / NB;
			    int h = det.rows / NB;
			    Mat hist;
			    for (int i=0;i<NB; i++) {
			        for (int j=0;j<NB; j++) {
			            Rect r(j*h, i*w, w, h);
			            r &= Rect(0,0,det.cols, det.rows);
			            lbp_xy(det, hist, r);
			        }
			    }
			    data.push_back(hist.reshape(1,1));
			    labels.push_back(f<5?0:(int)e);

            } else {
            	seq.push_back(det.clone());
            }
            if (useDiff) {
                imshow("dsum", dsum/(255));
            }
            imshow("I",det*10);
            waitKey(5);
        }
        if (singleImage)
        	continue;

        // mirror images in t, so peak expr is in the middle of seq.
        for (int i=(seq.size())-1; i>=0; i--) {
            seq.push_back(seq.at(i));
        }
        Mat hist = lbptop(seq);

        data.push_back(hist);
        labels.push_back((int)e);
    }
    cerr << '\n';
    cerr << "saving " << data.size() << " " << labels.size() << endl;
    FileStorage fs("ckplus_lbp.yml.gz",1);
    fs << "labels" << labels;
    fs << "data" << data;
    fs.release();
    cerr << Mat(histo).t() <<  endl;
    return 0;
}

/* best i can do, currently ;(
[0, 45, 18, 59, 25, 69, 28, 83] // class distribution
training with [944 x 4113] [1 x 4113]
testing with [944 x 1763]
accuracy: 0.719229
confusion:
[441, 96, 43, 63, 43, 43, 54, 52;
 14, 155, 4, 3, 0, 0, 29, 0;
 0, 0, 13, 0, 0, 0, 0, 0;
 2, 1, 0, 96, 0, 0, 0, 0;
 4, 0, 1, 0, 105, 0, 0, 0;
 14, 1, 0, 3, 3, 236, 0, 2;
 4, 1, 0, 0, 0, 0, 43, 0;
 10, 0, 0, 0, 2, 0, 3, 179] */
int main() {
    make_data(true,true);

    Mat data, labels;
    FileStorage fs("ckplus_lbp.yml.gz",0);
    fs["labels"] >> labels;
    fs["data"] >> data;
    fs.release();

    Ptr<ml::TrainData> tdata = ml::TrainData::create(data,0,labels);
    tdata->setTrainTestSplitRatio(0.7);
    data   = tdata->getTrainSamples();
    labels = tdata->getTrainResponses();
    labels.convertTo(labels, CV_32S); // hrrm!
    cerr << "training with " << data.size() << " " << labels.size() << endl;

    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setKernel(ml::SVM::LINEAR);
    svm->train(data,0,labels);

    Mat vdata = tdata->getTestSamples();
    Mat truth = tdata->getTestResponses();
    cerr << "testing with " << vdata.size() << endl;

    Mat predict;
    svm->predict(vdata, predict);
    predict.convertTo(predict, CV_32S); // hrrm!

    float correct = countNonZero(predict == truth);
    float accuracy = correct / predict.total();
    cerr << "accuracy: " << accuracy << endl;

    // accuracy alone is not enough here, since it might
    //   simply have missed all positives !
    Mat_<int> confusion(8,8,0);
    for (int i=0; i<predict.rows; i++) {
        int p = (int)predict.at<int>(i);
        int t = (int)truth.at<int>(i);
        confusion(p,t) ++;
    }
    cerr << "confusion:\n" << confusion << endl;
    return 0;
}
