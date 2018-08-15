#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"


#include <cstdio>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;

// (x-µ).t() * COV(-1) *(µ-x)
Mat pdf(const Mat &samp, const Mat &means, const Mat &covi) {
    Mat dx = samp - means;
    Mat p = -0.5 * (dx.reshape(1, samp.total()) * covi) * dx;
    Mat e; exp(p,e);
    normalize(e,e);
    return e;
}

int main( int argc, char ** argv )
{
    const int nimages    = 20;
    const int nclusters  = 3;
    const char *datapath = "c:/data/img/cache";


    Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
    Ptr<DescriptorMatcher> matcher = makePtr<BFMatcher>(NORM_L2);
    Ptr<ml::EM> gmm = ml::EM::create();
    gmm->setClustersNumber(nclusters);

    Mat train,weights,means;
    vector<Mat> covs;

    FileStorage fs1("fisher.yml",0|FileStorage::BASE64);
    if (! fs1.isOpened()) {
        vector<String> fn;
        glob(datapath,fn,true);
        cout << fn.size() << " images." << endl;

        for (int i=0; i<nimages; i++) {
            Mat im,feat;
            std::vector<KeyPoint> kp;
            int id = theRNG().uniform(0,fn.size());
            im = imread(fn[i],0);
            if (im.empty()) continue;
            if (im.rows>512) {
                resize(im,im,Size(), .25, .25);
            }
            f2d->detectAndCompute(im, Mat(), kp, feat);
            if (feat.empty()) continue;
            train.push_back(feat);
            cout << i << "\r";
        }

        gmm->train(train,0,noArray());
        means = gmm->getMeans();
        weights = gmm->getWeights();
        gmm->getCovs(covs);

        FileStorage fs("fisher.yml",1|FileStorage::BASE64);
        fs << "mean" << means;
        fs << "weights" << gmm->getWeights();
        fs << "covs" << covs;
        fs << "train" << train;
        fs.release();
    } else {
        fs1["mean"] >> means;
        fs1["weights"] >> weights;
        fs1["covs"] >> covs;
        fs1["train"] >> train;
        fs1.release();
    }

    cout << train.size() << " features." << endl;
    cout << "mean " << means.size() << endl;
    cout << "weig " << weights.size() << endl;
    cout << "covs " << covs.size() << " " << covs[0].size() << endl;

    return 0;
}
