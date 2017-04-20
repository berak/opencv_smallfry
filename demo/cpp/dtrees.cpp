
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
    Ptr<cv::ml::TrainData> myTrainData = cv::ml::TrainData::create(m, cv::ml::ROW_SAMPLE, l);

    /*
    Ptr<cv::ml::DTrees> dt = cv::ml::DTrees::create();
    dt->setMaxDepth(10);
    dt->setCVFolds(0);
    bool ok = dt->train(myTrainData);
    */

    Ptr<cv::ml::Boost> bo = cv::ml::Boost::create();
   // bo->setMaxDepth(10);
   // bo->setCVFolds(0);
    bo->setBoostType(cv::ml::Boost::REAL);
    bo->setWeakCount(100);
    bo->setWeightTrimRate(0.95);
    bo->setMaxDepth(1);
    bo->setUseSurrogates(false);
    bool ok = bo->train(myTrainData);


    return EXIT_SUCCESS;
}
