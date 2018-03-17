

    #include <iostream>
    #include <opencv2/opencv.hpp>
    #include <opencv2/core/core.hpp>
    #include <opencv2/highgui/highgui.hpp>
    #include <opencv2/ml.hpp>
    #include <opencv2/core.hpp>
    #include <opencv2/imgproc.hpp>
    #include "opencv2/imgcodecs.hpp"
    #include <opencv2/highgui.hpp>
    #include <opencv2/ml.hpp>

    using namespace cv::ml;
    using namespace cv;
    using namespace std;



    int main()
    {
        float labels[4] = {1, -1, -1, -1};
        float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
        Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
        Mat labelsMat(4, 1, CV_32F, labels);

        Ptr<SVM> svm = SVM::create();
        svm->setP( 0.1 );
        svm->setType(SVM::EPS_SVR );
        svm->setKernel(SVM::LINEAR);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
        cv::Ptr<cv::ml::TrainData> trainDataPtr = cv::ml::TrainData::create(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);
        bool ok = svm->trainAuto(trainDataPtr);
        cout << ok << endl;
        return 0;
    }
