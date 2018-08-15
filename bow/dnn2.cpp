#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/core/utils/filesystem.hpp>
#include "opencv2/dnn.hpp"

#include "profile.h"

#include <cstdio>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

string category(string s, int off) {
    string s2 = s.substr(off);
    int se = s2.find_last_of('\\');
    if (se < 0)
        se = s2.find_last_of('/');
    return s2.substr(0,se);
}

int main( int argc, char ** argv ) {
    const String datapath = "c:/data/caltech/101_ObjectCategories";
    String fname  = argc>1 ? argv[1] : "dnn";
    int nclusters = argc>2 ? atoi(argv[2]) : 64;
    int ncimages  = argc>3 ? atoi(argv[3]) : 2000; // can't handle more
    int ntimages  = argc>4 ? atoi(argv[4]) : 100;
    //theRNG().state = getTickCount();
    ocl::setUseOpenCL(false);

    vector<String> fn;
    glob(datapath + "/*.jpg", fn, true);
    cout << fn.size() << " (";
    for (auto a=fn.begin(); a!= fn.end();) {
        if(category(*a,datapath.size()+1) == "BACKGROUND_Google") {
            a = fn.erase(a);
        } else {
            a ++;
        }
    }
    cout << fn.size() << ") filenames." << endl;

    random_shuffle(fn.begin(), fn.end());
    map<String,int> cats;
    for (size_t i=0,k=0; i<fn.size(); i++) {
        String c = category(fn[i],datapath.size()+1);
        if (cats.find(c) == cats.end()) {
            cats[c] = k++;
        }
    }
    int max_classes = cats.size();

    Mat trainData, testData;
    FileStorage fs3(fname+".train.yml", 0);
    fs3["train"] >> trainData;
    fs3["test"] >> testData;
    fs3.release();
    testData = testData.reshape(1,ntimages);
    trainData = trainData.reshape(1,fn.size()-ntimages);

    Mat trainLabels(trainData.rows, cats.size(), CV_32F, 0.0f),
        svmLabels(trainData.rows, 1, CV_32S, 0.0f),
        testLabels(testData.rows, 1, CV_32F, 0.0f);
    for (int i=0; i<fn.size(); i++) {
        String cat = category(fn[i],datapath.size()+1);
        int id = cats[cat];
        if (i < ntimages)
            testLabels.at<float>(i,0) = id;
        else {
            trainLabels.at<float>(i-ntimages,id) = 1.0f;
            svmLabels.at<int>(i-ntimages,0) = id;
        }
    }
    cout << "train " << fname << " " << trainData.size() << " " << testData.size() << endl;
/*
    Mat_<int> layers(4, 1);
    layers << trainData.cols, 256, 128, cats.size();
    cout << "nn " << layers.t() << endl;
    Ptr<ml::ANN_MLP> nn = ml::ANN_MLP::create();
    nn->setLayerSizes(layers);
    nn->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0004);
    nn->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
    nn->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 300, 0.0001));
*/
    Ptr<ml::SVM> nn = ml::SVM::create();
    //nn->setKernel(ml::SVM::LINEAR);

    nn->train(trainData,0,svmLabels);

    float correct = 0;
    for (int r=0; r<testData.rows; r++) {
        float p = nn->predict(testData.row(r));
        float q = testLabels.at<float>(r);
        correct += p==q;
        cout << r << " " << p << " "<< q << " " << (p==q) << endl;
    }


    float acc = correct / ntimages;
    cout << "final " << correct << " / " << ntimages << " : " << acc << endl;

    return 0;
}
