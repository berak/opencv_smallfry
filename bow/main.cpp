#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "BowKMajorityTrainer.h"

#include <cstdio>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;

//
// cheesy test with the left/right opencv checkerboard samples.
//

int main( int argc, char ** argv )
{
    const int nimages    = 8;
    const int nclusters  = 100;
    const char *datapath = "c:/p/opencv/samples/data";


#if 0 // SIFT,SURF
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    Ptr<DescriptorMatcher> matcher = makePtr<BFMatcher>(NORM_L2);
    Ptr<BOWTrainer> bow_train = makePtr<BOWKMeansTrainer>(nclusters);
#else // ORB,BRISK,AKAZE, etc.
    Ptr<Feature2D> f2d = AKAZE::create();
    Ptr<DescriptorMatcher> matcher = makePtr<BFMatcher>(NORM_HAMMING);
    Ptr<BOWTrainer> bow_train = makePtr<BOWKmajorityTrainer>(nclusters, 10);
#endif
    Ptr<BOWImgDescriptorExtractor> bow_extract = makePtr<BOWImgDescriptorExtractor>(f2d,matcher);

    //
    // Step 1: gather images, and train BOW using Feature2D features.
    //  in real life ofc. this data should be totally *seperate*
    //  from your svm train / test data !
    //
    for (int i=0; i<nimages; i++)
    {
        Mat im[2],feat[2];
        std::vector<KeyPoint> kp[2];

        im[0] = imread(format("%s/left%02d.jpg",datapath,i+1),0);
        f2d->detectAndCompute(im[0], Mat(), kp[0], feat[0]);
        bow_train->add(feat[0]);

        im[1] = imread(format("%s/right%02d.jpg",datapath,i+1),0);
        f2d->detectAndCompute(im[1], Mat(), kp[1], feat[1]);
        bow_train->add(feat[1]);
    }
    cout << bow_train->descriptorsCount() << " features." << endl;
    Mat vocab = bow_train->cluster();
    bow_extract->setVocabulary(vocab);

    cerr << "bow vocab: " << vocab.size() << endl;
    //cerr << vocab << endl;

    //
    // Step 2: gather SVM train data using BOWImgDescriptorExtractor
    //
    Mat svmTrainData;
    Mat svmTrainLabels;
    for (int i=2; i<nimages; i++) // save 1st 2*2 images for testing later
    {
        Mat im[2],bowfeat[2];
        std::vector<KeyPoint> kp[2];

        im[0] = imread(format("%s/left%02d.jpg",datapath,i+1),0);
        f2d->detect(im[0],kp[0]);
        bow_extract->compute2(im[0], kp[0], bowfeat[0]);
        svmTrainData.push_back(bowfeat[0]);
        svmTrainLabels.push_back(1);

        im[1] = imread(format("%s/right%02d.jpg",datapath,i+1),0);
        f2d->detect(im[1],kp[1]);
        bow_extract->compute2(im[1], kp[1], bowfeat[1]);
        svmTrainData.push_back(bowfeat[1]);
        svmTrainLabels.push_back(-1);
    }
    cerr << "svm data:  " << svmTrainData.size() << endl;

    //
    // Step 3: train the svm:
    //

    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setKernel(ml::SVM::INTER);
    svm->setC(50);
    //svm->setGamma(0.8);

    bool ok = svm->train(ml::TrainData::create(svmTrainData, ml::ROW_SAMPLE, svmTrainLabels));
    cerr << "svm train : " << ok << endl;

    //
    // Step 4: predict the remaining 2*2 images, use BOWImgDescriptorExtractor again
    //
    for (int i=0; i<2; i++)
    {
        Mat im[2],bowfeat[2];
        std::vector<KeyPoint> kp[2];

        im[0] = imread(format("%s/left%02d.jpg",datapath,i+1),0);
        f2d->detect(im[0],kp[0]);
        bow_extract->compute2(im[0], kp[0], bowfeat[0]);
        cerr << "left  : " << svm->predict( bowfeat[0] ) << endl;

        im[1] = imread(format("%s/right%02d.jpg",datapath,i+1),0);
        f2d->detect(im[1],kp[1]);
        bow_extract->compute2(im[1], kp[1], bowfeat[1]);
        cerr << "right : " << svm->predict( bowfeat[1] ) << endl;
    }
    return 0;
}
