
#include <opencv2/opencv.hpp>
#include <iostream>
#include <deque>

using namespace cv;
using namespace std;


int main()
{
    Mat m(100,100,CV_32F);
    randu(m,0,255);
    Mat l(100,1,CV_32S);
    randu(m,0,2);

    Ptr<cv::ml::DTrees> model = cv::ml::DTrees::create();
    model->setMaxDepth(10);
    model->setMinSampleCount(2);
    //m->setRegressionAccuracy(REG_ACCURACY);
    //model->setUseSurrogates(true);
    //model->setMaxCategories(10);
    model->setCVFolds(0);
    //model->setUse1SERule(true);
    //m->setTruncatePrunedTree(IS_PRUNED);
    //model->setPriors(Mat());


    Ptr<cv::ml::TrainData> myTrainData = cv::ml::TrainData::create(m, cv::ml::ROW_SAMPLE, l);
    bool ok = model->train(myTrainData);
    return EXIT_SUCCESS;
}
