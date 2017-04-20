#include "opencv2/opencv.hpp"
#include "opencv2/tracking.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setKernel(ml::SVM::CHI2);

    Mat_<float> data(4,2);
    data << 1,1,  1.2,0.9,  3.3,2.8, 3.1,3.6;
    Mat_<int> labels(4,1);
    labels << -1,-1, 1,1;
    svm->train(data,0,labels);

    Mat_<float> query(4,2);
    query << 0.8,1,  1.5,1.3,  3.9,2.5, 2.9,3.1;
    Mat result;
    svm->predict(query, result, 0); // use ml::StatModel::RAW_OUTPUT for the raw dot product;
    cerr << result.t() << endl;
    return 0;
}
