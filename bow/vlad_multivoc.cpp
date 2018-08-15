#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/core/utils/filesystem.hpp>
#include "opencv2/xfeatures2d.hpp"

#include "BowKMajorityTrainer.h"
#include "profile.h"

#include <cstdio>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

int missedImg=0;
int missedFeat=0;

static Mat rootsift(Mat &feat) {
    PROFILE
    normalize(feat,feat,NORM_L1);
    sqrt(feat,feat);
    patchNaNs(feat, 0); // for SURF(which has neg. numbers)
    return feat;
}

static Mat getImg(String fn, String fname) {
    PROFILE
    Mat img = imread(fn, fname!="lucid"?0:1);
    if (! img.empty()) {
        if (img.rows>512) {
            resize(img,img,Size(), .25, .25);
        }
    } else {
        missedImg += 1;
        cout << "noimg " << fn << endl;
    }
    return img;
}

static Mat sign(const Mat &m) {
    PROFILE
    Mat s(m.size(),m.type(),1.0f);
    s.setTo(-1, m<0);
    return s;
}

static Mat vlad_feature(Ptr<Feature2D> f2d, Ptr<DescriptorMatcher> matcher, const vector<Mat> &vocabs, const Mat &img) {
    PROFILE
    std::vector<KeyPoint> kp;
    Mat feat, desc;
    {
        PROFILEX("vlad:detect")
        f2d->detectAndCompute(img, Mat(), kp, desc);
    }
    if (desc.rows>0) {
        PROFILEX("vlad:compute")
        rootsift(desc);

        Mat feat;
        for (size_t v=0; v<vocabs.size(); v++) {
            vector<DMatch> matches;
            matcher->match(desc, vocabs[v], matches);
            Mat f = Mat(vocabs[v].size(), CV_32F, 0.0f);
            {
                PROFILEX("vlad:vlad")
                for (size_t j=0; j<matches.size(); j++) {
                    Mat dr = desc.row(matches[j].queryIdx);
                    Mat vr = vocabs[v].row(matches[j].trainIdx);
                    Mat re = vr - dr;
                    normalize(re,re); // innorm
                    f.row(matches[j].trainIdx) += re / vocabs[v].rows;
                }
            }
            feat.push_back(f);
        }
        {
            PROFILEX("vlad:post")
            // power normalization
            Mat f2;
            sqrt(abs(feat), f2);
            feat = sign(feat).mul(f2);
            // L2 normalization
            normalize(feat,feat);
        }
        return feat.reshape(1,1);
    }
    missedFeat ++;
    return Mat();
}

static Mat vlad_feature(Ptr<Feature2D> f2d, Ptr<DescriptorMatcher> matcher, const Mat &vocab, const Mat &img) {
    PROFILE
    std::vector<KeyPoint> kp;
    Mat desc;
    {
        PROFILEX("vlad:detect")
        f2d->detectAndCompute(img, Mat(), kp, desc);
    }
    if (desc.rows>0) {
        Mat feat = Mat(vocab.size(), CV_32F, 0.0f);
        {
            PROFILEX("vlad:compute")
            rootsift(desc);

            vector<DMatch> matches;
            matcher->match(desc, vocab, matches);
            {
                PROFILEX("vlad:vlad")
                for (size_t j=0; j<matches.size(); j++) {
                    Mat dr = desc.row(matches[j].queryIdx);
                    Mat vr = vocab.row(matches[j].trainIdx);
                    Mat re = vr - dr;
                    normalize(re,re); // innorm
                    feat.row(matches[j].trainIdx) += re / vocab.rows;
                }
            }
        }
        {
            PROFILEX("vlad:post")
            // power normalization
            Mat f2;
            sqrt(abs(feat), f2);
            feat = sign(feat).mul(f2);
            // L2 normalization
            normalize(feat,feat);
        }
        return feat.reshape(1,1);
    }
    missedFeat ++;
    return Mat();
}


static cv::Ptr<cv::flann::Index> train_index(const Mat &trainData) {
    PROFILE
    return makePtr<cv::flann::Index>(trainData, cv::flann::LinearIndexParams(), cvflann::FLANN_DIST_L2);
}

string category(string s, int off) {
    string s2 = s.substr(off);
    int se = s2.find_last_of('\\');
    if (se < 0)
        se = s2.find_last_of('/');
    return s2.substr(0,se);
}

int main( int argc, char ** argv ) {
    const String datapath = "c:/data/caltech/101_ObjectCategories";
    //const String datapath = "c:/data/img/cache";
    String fname  = argc>1 ? argv[1] : "vlad";
    int nclusters = argc>2 ? atoi(argv[2]) : 64;
    int ncimages  = argc>3 ? atoi(argv[3]) : 6000;
    int ntimages  = argc>4 ? atoi(argv[4]) : 100;
    //theRNG().state = getTickCount();
    ocl::setUseOpenCL(false);

    Ptr<Feature2D> det = xfeatures2d::SURF::create(100,4,3,false,true);;
    Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>();;
    Ptr<BOWTrainer> bow_train = makePtr<BOWKMeansTrainer>(nclusters);;

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

    // 1. build dictionary
    Mat vocab1;
    FileStorage fs(fname+".voc.yml", 0);
    fs["voc"] >> vocab1;
    fs.release();

    Mat vocab2;
    FileStorage fs2(fname+".voc1.yml", 0);
    fs2["voc"] >> vocab2;
    fs2.release();
//    vector<Mat> vocabs {vocab1, vocab2};
    vocab1.push_back(vocab2);
    cerr << "bow vocab: " << vocab1.size() << " " << vocab2.size() << endl;

    int fsize=0;
    Mat trainData;

    // 2. build train data for index
    FileStorage fs1(fname+".dat1.yml", 0);
    if (!fs1.isOpened()) {
        for (int i=ntimages; i<fn.size(); i++) {
            PROFILEX("train data")

            int id = i; //theRNG().uniform(0,fn.size());

            Mat img = getImg(fn[id],fname);
            if (img.empty()) continue;

            Mat feat = vlad_feature(det, matcher, vocab1, img);
            if (feat.rows) {
                trainData.push_back(feat);
            }
            fsize += feat.rows;
            cout << i << "\r";
            if (i%100==99)
                cout << "train " << fname << " " << id << " " << feat.size() << " " << fsize << endl;
        }
        FileStorage fs2(fname+".dat1.yml", 1+FileStorage::BASE64);
        fs2 << "dat" << trainData;
        fs2.release();
    } else {
        fs1["dat"] >> trainData;
        fs1.release();
    }
    cout << "train " << fname << " " << trainData.size() << " " << trainData.type() << endl;

    // 3. build the index
    Ptr<cv::flann::Index> index;
    index = train_index(trainData);
    cout << "index " << trainData.size() << " " << trainData.type() << endl;

    // 4. run tests
    int K=5;
    float correct=0; int ntests=0;
    cv::flann::SearchParams params;
    for (int i=0; i<ntimages; i++) {
        int id = i;
        String cat = category(fn[id], datapath.size() + 1) ;
        if (cat == "faces_easy") cat = "faces";

        Mat org = getImg(fn[id],fname);
        if (org.empty()) continue;

        Mat feat = vlad_feature(det, matcher, vocab1, org);
        if (feat.empty()) continue;
        cv::Mat dists;
        cv::Mat found;
        index->knnSearch(feat, found, dists, K, params);
        cout << "test " << cat << " " << found << endl;

        Mat res;
        resize(org,res,Size(240,160));
        for (int j=0; j<K; j++) {
            int id = found.at<int>(0,j);
            Mat img = imread(fn[id],fname!="lucid"?0:1);
            if (img.empty()) continue;
            String cat2 = category(fn[id], datapath.size() + 1) ;
            if (cat2=="faces_easy") cat2="faces";
            bool ok = (cat == cat2);
            cout << "   " << ok << " " << cat2 << endl;
            correct += ok;
            ntests ++;
            resize(img,img,Size(240,160));
            hconcat(res,img,res);
        }
        imshow("I",res);
        waitKey(3000);
    }

    float acc = correct / ntests;
    cout << "final " << correct << " / " << ntests << " : " << acc << endl;
    cout << "missed " << missedImg << " images and " << missedFeat << " features." << endl;

    return 0;
}
