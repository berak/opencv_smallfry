
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <opencv2/face/bif.hpp>

#include <iostream>

using namespace cv;
using namespace std;

float train_test_ann(int nclasses, const Mat &train_data, const Mat &train_labels, const Mat &test_data, const Mat &test_labels, Mat &confusion) {
    // setup the ann:
    int nfeatures = train_data.cols;
    Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::create();
    Mat_<int> layers(4,1);
    layers(0) = nfeatures;     // input
    layers(1) = nclasses * 8;  // hidden
    layers(2) = nclasses * 4;  // hidden
    layers(3) = nclasses;      // output, 1 pin per class.
    ann->setLayerSizes(layers);
    ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM,0,0);
    ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 300, 0.0001));
    ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);

    // ann requires "one-hot" encoding of class labels:
    Mat train_classes = Mat::zeros(train_data.rows, nclasses, CV_32FC1);
    for(int i=0; i<train_classes.rows; i++)
    {
        train_classes.at<float>(i, train_labels.at<int>(i)) = 1.f;
    }
    //cerr << train_data.size() << " " << train_classes.size() << endl;

    ann->train(train_data, ml::ROW_SAMPLE, train_classes);

    // run tests on validation set:
    for(int i=0; i<test_data.rows; i++) {
        int pred  = ann->predict(test_data.row(i), noArray());
        int truth = test_labels.at<int>(i);
        confusion.at<int>(pred, truth) ++;
    }
    Mat correct = confusion.diag();
    float accuracy = sum(correct)[0] / sum(confusion)[0];
    cerr << "accuracy: " << accuracy << endl;
    //cerr << "confusion:\n" << confusion << endl;
    return accuracy;
}

float train_test_lbph(int nclasses, const Mat &train_data, const Mat &train_labels, const Mat &test_data, const Mat &test_labels, Mat &confusion) {
    Ptr<face::FaceRecognizer> fr = cv::face::LBPHFaceRecognizer::create();
    vector<Mat> td;
    for (int n=0; n<train_data.rows; n++) {
        td.push_back(train_data.row(n).reshape(1,96));
    }
    fr->train(td, train_labels);

    // run tests on validation set:
    for(int i=0; i<test_data.rows; i++) {
        int pred  = fr->predict(test_data.row(i).reshape(1,96));
        int truth = test_labels.at<int>(i);
        confusion.at<int>(pred, truth) ++;
    }
    Mat correct = confusion.diag();
    float accuracy = sum(correct)[0] / sum(confusion)[0];
    cerr << "accuracy:  " << accuracy << endl;
    //cerr << "confusion:\n" << confusion << endl;
    return accuracy;
}

int main(int argc, char** argv) {
    String att = "c:/data/faces/att/*.pgm"; //40 persons a 10 images
    String lfw = "c:/data/faces/lfw40_crop/*.jpg"; //40 persons a 10 images
    int method = argc>1 ? atoi(argv[1]) : 0;
    enum {
        FACENET,
        BIF,
        LBPH
    };
    String ds = argc>2 ? att : lfw;
    vector<String> fn;
    glob(ds,fn,true);


    int rot = 4; // BIF
    int bands = 6; //BIF
    int nclasses = 40; // of 40, i only got 2gb mem ..
    Mat confusion(nclasses,nclasses,CV_32S, Scalar(0)); // will hold our test results

    Ptr<cv::face::BIF> bif;
    if (method==BIF)
        bif = cv::face::BIF::create(bands, rot);

    dnn::Net net;
    if (method==FACENET)
        net = dnn::readNetFromTorch("c:/data/mdl/openface.nn4.small2.v1.t7");

    int off = 0;
    int split = 2;
    float mean_acc = 0;
    for (int f=0; f<5; f++) { // folds
        Mat train_data,train_labels, test_data,test_labels;
        for (int p=0; p<nclasses; p++) { // persons
            cerr << "p " << p << "\r";
            for (int i=0; i<10; i++) {
                int k = p * 10 + i; // + 1; // att image index starts from 1
                cv::Mat image = cv::imread(fn[k], (method==FACENET) ? 1 : 0);
                if (image.empty()) {cerr << "no !" << fn[k] << endl; continue; }
                Mat feature;
                if (method==FACENET) {
                    Mat inputBlob = dnn::blobFromImage(image, 1./255, Size(96,96), Scalar(), true, false);   //Convert Mat to image batch
                    net.setInput(inputBlob);
                    feature = net.forward();
                } else if (method==BIF) {
                    image.convertTo(image, CV_32F, 1.0/255);
                    bif->compute(image, feature);
                } else {
                    resize(image, feature, Size(96,96));
                }
                // split into train / test folds:
                if ((i>=off) && (i<off+split)) {
                    test_data.push_back(feature.reshape(1,1)); // flatten to 1 row
                    test_labels.push_back(p);
                } else {
                    train_data.push_back(feature.reshape(1,1));
                    train_labels.push_back(p);
                }
            }
        }
        static int once=1;
        if (once) {
            cout << train_data.size() << " " << ds << endl;
            once=0;
        }
        cerr << "fold " << f << " ";
        if (method==LBPH) {
            mean_acc += train_test_lbph(nclasses, train_data, train_labels, test_data, test_labels, confusion);
        } else {
            mean_acc += train_test_ann(nclasses, train_data, train_labels, test_data, test_labels, confusion);
        }
        off += split;
    }
    cout << ((method==FACENET) ? "FACENET ": (method==BIF) ? "BIF ": "lbph " );
    cout << "accuracy: " << mean_acc / 5 << endl;
    //cerr << "confusion:\n" << confusion << endl;
    return 0;
}


/*
$> facenet 0
[128 x 320] c:/data/faces/lfw40_crop/*.jpg
fold 0 accuracy: 0.7125
fold 1 accuracy: 0.71875
fold 2 accuracy: 0.695833
fold 3 accuracy: 0.690625
fold 4 accuracy: 0.7025
FACENET accuracy: 0.704042

$> facenet 1
[10384 x 320] c:/data/faces/lfw40_crop/*.jpg
fold 0 accuracy: 0.65
fold 1 accuracy: 0.55
fold 2 accuracy: 0.529167
fold 3 accuracy: 0.559375
fold 4 accuracy: 0.555
BIF accuracy: 0.568708

$> facenet 2
[9216 x 320] c:/data/faces/lfw40_crop/*.jpg
fold 0 accuracy:  0.6375
fold 1 accuracy:  0.63125
fold 2 accuracy:  0.620833
fold 3 accuracy:  0.61875
fold 4 accuracy:  0.625
lbph accuracy: 0.626667


(it's somewhat unfair, att is grayscale ...)
$> facenet 0 1
[128 x 320] c:/data/faces/att/*.pgm
fold 0 accuracy: 0.8125
fold 1 accuracy: 0.8
fold 2 accuracy: 0.795833
fold 3 accuracy: 0.79375
fold 4 accuracy: 0.7825
FACENET accuracy: 0.796917

$> facenet 1 1
[11820 x 320] c:/data/faces/att/*.pgm
fold 0 accuracy: 0.9875
fold 1 accuracy: 0.99375
fold 2 accuracy: 0.995833
fold 3 accuracy: 0.99375
fold 4 accuracy: 0.985
BIF accuracy: 0.991167

$> facenet 2 1
[9216 x 320] c:/data/faces/att/*.pgm
fold 0 accuracy:  0.9625
fold 1 accuracy:  0.96875
fold 2 accuracy:  0.970833
fold 3 accuracy:  0.971875
fold 4 accuracy:  0.965
lbph accuracy: 0.967792

*/
