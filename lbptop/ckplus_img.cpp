#include "opencv2/opencv.hpp"
#include <iostream>
#include "lbptop.h"

using namespace cv;
using namespace std;

int make_data() {
    CascadeClassifier cad("haarcascade_frontalface_alt.xml");
    String path = "c:/data/faces/ckplus/";
    vector<String> emos; // only 327 of all sequences are labelled, we have to check those 1st.
    glob(path+"Emotion/*.txt", emos, true);
    cerr << emos.size() << endl;
    vector<float> histo(8,0);

    FileStorage fs("ckplus_img.yml.gz",1);
    fs << "sequences" << "[";
    for (size_t z=0; z<emos.size(); z++) {
        String fn = emos[z];
        ifstream emo(fn.c_str());
        float e;
        emo >> e;
        histo[(int)e] ++;
        int l = fn.find_last_of('\\');
        String txt = fn.substr(l + 1);
        String sub = txt.substr(0, 4);
        String em  = txt.substr(5, 3);
        String frm = txt.substr(9, 8);
        cerr << e << " " << sub << " " << em << " " << frm << '\r';
        String imgbase = path + "cohn-kanade-images/" + sub + "/" + em + "/";
        int nfrm = stoi(frm);
        Rect box;
        Mat first;
        Mat labels;
        fs << "{";
        fs << "points" << "[";
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
            resize(gray(box), det, Size(64,64));
            det.convertTo(det, CV_32F);
            Scalar m,d;
            meanStdDev(det, m,d);
            det -= m[0];
            det /= d[0];

            fs << det;
			labels.push_back(f<nfrm/4 ? 0 : (int)e);

            imshow("I",det);
            waitKey(5);
        }
        fs << "]";
        fs << "labels" << labels;
        fs << "}";
    }
    cerr << '\n';
    fs << "]";
    fs.release();
    cerr << Mat(histo).t() <<  endl;
    return 0;
}

/*
[0, 45, 18, 59, 25, 69, 28, 83]
training with [4096 x 4113] [1 x 4113]
testing with [4096 x 1763]
accuracy: 0.859898
confusion:
[344, 21, 7, 59, 7, 35, 16, 39;
 8, 223, 0, 0, 0, 0, 0, 0;
 3, 0, 69, 0, 0, 0, 0, 0;
 5, 0, 0, 140, 0, 3, 0, 0;
 3, 0, 0, 0, 140, 0, 0, 1;
 13, 0, 0, 0, 0, 254, 0, 0;
 6, 0, 0, 0, 0, 0, 114, 0;
 20, 0, 0, 0, 0, 1, 0, 232]
*/
int main() {
    //make_data();

    Mat data, labels;
    FileStorage fs("ckplus_img.yml.gz",0);

    FileNode n = fs["sequences"];
    FileNodeIterator si = n.begin();
    for (; si != n.end(); ++si) {
        FileNode p = (*si)["points"];
        FileNodeIterator pi = p.begin();
        for (; pi != p.end(); ++pi) {
            Mat m;
            (*pi) >> m;
            data.push_back(m.reshape(1,1));
        }
        Mat l;
        (*si)["labels"] >> l;
        labels.push_back(l);
    }
    fs.release();
    labels.reshape(1,labels.total());
    data.convertTo(data, CV_32F);
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
