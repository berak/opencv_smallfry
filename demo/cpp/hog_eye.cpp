#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>
#include <iostream>
using namespace cv;
using namespace std;


float traintest(Ptr<ml::StatModel> mdl, const Mat &data, const Mat &labels, float ratio) {
    Ptr<ml::TrainData> tdata = ml::TrainData::create(data,0,labels);
    tdata->setTrainTestSplitRatio(ratio, true);

    Mat trn_d = tdata->getTrainSamples();
    Mat trn_l = tdata->getTrainResponses();
    trn_l.convertTo(trn_l, labels.type()); // hrrm!
    mdl->train(trn_d, 0, trn_l);

    Mat tst_d = tdata->getTestSamples();
    Mat tst_l = tdata->getTestResponses();
    Mat res;
    mdl->predict(tst_d, res);
    res.convertTo(res, tst_l.type());
    int correct = countNonZero(res==tst_l);
    float acc = float(correct) / float(res.total());
    return acc;
}

cv::Mat loadImg(const cv::String &path)
{
    cv::Mat im = cv::imread(path, 0);
    cv::equalizeHist(im,im);
    return im;
}
void compute_hog_dir(const cv::String &path, cv::Mat &gradients, const cv::Size & size)
{
    cv::HOGDescriptor hog;
    hog.winSize = size;
    vector< cv::Point > location;

    vector<cv::String> fn;
    glob(path,fn);

    for(size_t i=0; i<fn.size(); i++)
    {
        cv::Mat gray = loadImg(fn[i]);
        if (gray.empty())
            continue;
        vector< float > descriptors;
        hog.compute(gray, descriptors, cv::Size( 8, 8 ), cv::Size( 0, 0 ), location);
        gradients.push_back(cv::Mat(descriptors).clone().reshape(1,1));
    }
}

void process_one_side(const cv::String &pos, const cv::String &neg, const cv::Size &SIZ, cv::Mat &grads, cv::Mat &labels)
{
    // compute pos/neg hogs:
    compute_hog_dir(pos, grads, SIZ);
    int npos = grads.rows;

    compute_hog_dir(neg, grads, SIZ);
    int nneg = grads.rows - npos;

    cerr << npos << "\t" << pos << endl;
    cerr << nneg << "\t" << neg << endl;

    // setup svm labels (1 for open, -1 for closed):
    labels = cv::Mat(grads.rows, 1, CV_32S, cv::Scalar(-1));
    labels(cv::Range(0,npos), cv::Range::all()) = 1;
}



int main()
{
    float train_test_ratio = 0.7f;
    Mat grads,labels;
    process_one_side("c:/data/eyes/openLeftEyes/*.jpg", "c:/data/eyes/closedLeftEyes/*.jpg", Size(24,24), grads,labels);

    Ptr<ml::SVM> mdl = ml::SVM::create();
    float acc = traintest(mdl,grads,labels,train_test_ratio);
    cerr << "SVM : " << acc << endl;

    labels.convertTo(labels, CV_32F);
    Ptr<ml::SVMSGD> mdl2 = ml::SVMSGD::create();
    float acc2 = traintest(mdl2,grads,labels,train_test_ratio);
    cerr << "SGD : " << acc2 << endl;

    // train_one_side("E:/MEDIA/eyes/openRightEyes/*.jpg","E:/MEDIA/eyes/closedRightEyes/*.jpg","eyesRight.xml");
    return 0;
}
