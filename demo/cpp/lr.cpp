
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace cv::ml;
using namespace std;

int main(int argc, char** argv) {
Mat data(30, 1, CV_32S);
Mat labels(30, 1, CV_32S);
Mat test(30, 1, CV_32S);
// make training and testing data
for (int i = 0; i < 30; i++) {
    data.at<int>(i) = i % 10;
    test.at<int>(i) = i % 10;
    if (i % 10 < 5) {
        labels.at<int>(i) = 0;
    }
    else {
        labels.at<int>(i) = 1;
    }
}
cout << data.t() << endl;
cout << labels.t() << endl;

data.convertTo(data, CV_32F);
labels.convertTo(labels, CV_32F);

cout << "Training samples:" << data.rows << ", Testing samples:" << test.rows << endl;
cout << "Training..." << endl;

//Ptr<SVM> lr1 = SVM::create();
Ptr<LogisticRegression> lr1 = LogisticRegression::create();
lr1->setLearningRate(0.01);
lr1->setIterations(10001);
lr1->setMiniBatchSize(10);

//lr1->setRegularization(LogisticRegression::REG_DISABLE);
lr1->setTrainMethod(LogisticRegression::MINI_BATCH);

lr1->train(data, ROW_SAMPLE, labels);
cout << "Testing..." << endl;

Mat responses;
test.convertTo(test, CV_32F);
cout << test.t() << endl;
lr1->predict(test, responses);
cout << "Done!" << endl << endl;

cout << "Original vs Predicted:" << endl;
//labels.convertTo(labels, CV_32S);
cout << labels.t() << endl;
cout << responses.t() << endl;
    return 0;
}
