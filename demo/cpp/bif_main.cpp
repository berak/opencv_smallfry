
#include <opencv2/opencv.hpp>

// uncomment, if you want to use BIF (opencv_contrib, 3.1)
//#include "opencv2/face/bif.hpp"

#include <iostream>

using namespace cv;
using namespace std;

void train_test(int nclasses, const Mat &train_data, const Mat &train_labels, const Mat &test_data, const Mat &test_labels, Mat &confusion) {
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
    cerr << train_data.size() << " " << train_classes.size() << endl;

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
    cerr << "confusion:\n" << confusion << endl;
}

int main(int argc, char** argv) {
    int rot = 4; // BIF
    int bands = 6; //BIF
    int nclasses = 10; // of 40, i only got 2gb mem ..
    String att = "c:/data/faces/att/"; //40 persons a 10 images
    vector<String> fn;
    glob(att,fn,true);

    Mat confusion(nclasses,nclasses,CV_32S, Scalar(0)); // will hold our test results

    // uncomment, if you want BIF !
    //Ptr<cv::face::BIF> bif = cv::face::createBIF(rot,bands);

    int off = 0;
    int split = 2;
    for (int f=0; f<5; f++) { // folds
        cerr << "fold " << f << endl;
        Mat train_data,train_labels, test_data,test_labels;
        for (int p=0; p<nclasses; p++) { // persons
            cerr << "p " << p << "\r";
            for (int i=0; i<10; i++) {
                int k = p * 10 + i + 1; // att image index starts from 1
                cv::Mat image = cv::imread(fn[k], 0);
                if (image.empty()) {cerr << "no !" << fn[k] << endl; continue; }

                image.convertTo(image, CV_32F);//1.0/255);
                resize(image, image, Size(80, 80));

                Mat feature = image;
                // uncomment, if you want BIF !
                //bif->compute(image, feature);

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

        train_test(nclasses, train_data, train_labels, test_data, test_labels, confusion);
        off += split;
    }
    return 0;
}
