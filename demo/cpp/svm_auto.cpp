#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

bool trainSVM(const cv::Mat &samples, vector<int> &labels) {
    CV_Assert(!samples.empty() && samples.type() == CV_32FC1);
    CV_Assert(!labels.empty());
    Ptr<ml::SVM> outSVM = cv::ml::SVM::create();
    outSVM->setType(ml::SVM::C_SVC);
    outSVM->setKernel(ml::SVM::CHI2);
    Ptr<ml::TrainData> pt = ml::TrainData::create(samples, ml::ROW_SAMPLE, labels);
    bool result = outSVM->trainAuto(pt, 2);
    outSVM->save("svm.xml");
    return result;
}

int main(int argc, char** argv) {

 FileStorage f1("samples.yml",0);
 Mat samples;
 f1["samples"] >> samples;
 f1.release();

FileStorage f2("labels.yml",0);
 vector<int> labels;
 f2["labels"] >> labels;
 f2.release();
    trainSVM(samples, labels);
    /*Ptr<ml::SVM> outSVM = cv::ml::SVM::create();
    outSVM->setType(ml::SVM::C_SVC);
    outSVM->setKernel(ml::SVM::CHI2);
    //outSVM->setC(100);
    //outSVM->setGamma(100);

    Mat samples(200,200,CV_32F);
    randu(samples,0,1);
    Mat labels(200,1,CV_32S);
    randu(labels,0,5);
    Ptr<ml::TrainData> pt = ml::TrainData::create(samples, ml::ROW_SAMPLE,
            labels);
    // bool result = outSVM->train(pt);
    bool result = outSVM->trainAuto(pt);
    cout << result << endl;
    */

    return 0;
}