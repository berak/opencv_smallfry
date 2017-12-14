#include "opencv2/opencv.hpp"
#include <iostream>
#include "lbptop.h"

using namespace cv;
using namespace std;
//#include "landmarks.h"
#if 0
int make_data() {
    CascadeClassifier cad("haarcascade_frontalface_alt.xml");
    Ptr<Landmarks> lm = createLandmarks();
    String path = "c:/data/faces/ckplus/";
    vector<String> emos; // only 327 of all sequences are labelled, we have to check those 1st.
    glob(path+"Emotion/*.txt", emos, true);
    cerr << emos.size() << endl;
    vector<float> histo(8,0);

    FileStorage fs("ckplus_lm.yml.gz",1);
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
            resize(gray(box), det, Size(200,200));
            vector<Point2f> points;
            lm->extract(det, points);
			Mat mm(Mat(points).reshape(1,1));
            //cout << mm << endl;
            fs << mm;
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
#endif

/* best i can do, currently ;(
327 1 0
[0, 45, 18, 59, 25, 69, 28, 83] // class distribution
training with [136 x 4113] [1 x 4113]
testing with [136 x 1763]
accuracy: 0.774248
confusion:
[358, 48, 38, 57, 21, 39, 41, 57;
 13, 192, 2, 7, 0, 0, 11, 1;
 0, 0, 31, 0, 0, 0, 0, 0;
 7, 2, 0, 132, 0, 0, 2, 1;
 8, 0, 0, 1, 117, 0, 2, 0;
 7, 0, 5, 1, 2, 253, 0, 0;
 5, 2, 0, 1, 0, 0, 71, 2;
 4, 0, 0, 0, 7, 1, 3, 211]
*/
int main() {
  //  make_data();

    Mat data, labels;
    FileStorage fs("ckplus_lm.yml.gz",0);

    FileNode n = fs["sequences"];
    FileNodeIterator si = n.begin();
    for (; si != n.end(); ++si) {
        FileNode p = (*si)["points"];
        FileNodeIterator pi = p.begin();
        for (; pi != p.end(); ++pi) {
            Mat m;
            (*pi) >> m;
            Scalar M,D;
            meanStdDev(m,M,D);
            m -= M[0];
            m /= D[0];
         //   cout << m << endl;
            data.push_back(m.reshape(1,1));
        }
        Mat l;
        (*si)["labels"] >> l;
       // cout << l << endl;
        labels.push_back(l);
    }
    fs.release();
    labels.reshape(1,labels.total());

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
