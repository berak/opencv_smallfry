#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "BowKMajorityTrainer.h"
#include "profile.h"

#include <cstdio>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;


static void trainBinary(Ptr<DescriptorMatcher> &matcher, Ptr<BOWTrainer> &bow_train, int nclusters) {
    matcher = makePtr<BFMatcher>(NORM_HAMMING);
    bow_train = makePtr<BOWKmajorityTrainer>(nclusters, 10);
}

static void trainFloat(Ptr<DescriptorMatcher> &matcher, Ptr<BOWTrainer> &bow_train, int nclusters) {
    matcher = makePtr<FlannBasedMatcher>();
    bow_train = makePtr<BOWKMeansTrainer>(nclusters);
}

static void trainFeatures(const String &fname, Ptr<Feature2D> &det, Ptr<Feature2D> &ext, Ptr<DescriptorMatcher> &matcher, Ptr<BOWTrainer> &bow_train, int nclusters) {
   if (fname=="surf")    { det = ext = xfeatures2d::SURF::create(); trainFloat(matcher, bow_train, nclusters); }
   if (fname=="sift")    { det = ext = xfeatures2d::SIFT::create(); trainFloat(matcher, bow_train, nclusters); }
   if (fname=="vlad")    { det = ext = xfeatures2d::SURF::create(); trainFloat(matcher, bow_train, nclusters); }
   if (fname=="latch")   { det = ORB::create(); ext = xfeatures2d::LATCH::create(); trainBinary(matcher, bow_train, nclusters); }
   if (fname=="lucid")   { det = ORB::create(); ext = xfeatures2d::LUCID::create(3); trainBinary(matcher, bow_train, nclusters); }
   if (fname=="vgg")     { det = xfeatures2d::SURF::create(); ext = xfeatures2d::VGG::create(); trainFloat(matcher, bow_train, nclusters); }
   if (fname=="boost")   { det = xfeatures2d::SURF::create(); ext = xfeatures2d::BoostDesc::create(); trainBinary(matcher, bow_train, nclusters); }
   if (fname=="orb")     { det = ext = ORB::create(); trainBinary(matcher, bow_train, nclusters); }
   if (fname=="brisk")   { det = ORB::create(); ext = BRISK::create(); trainBinary(matcher, bow_train, nclusters); }
   if (fname=="brief")   { det = ORB::create(); ext = xfeatures2d::BriefDescriptorExtractor::create(); trainBinary(matcher, bow_train, nclusters); }
}

static Mat getImg(String fn, String fname) {
    Mat img = imread(fn, fname!="lucid"?0:1);
    if (! img.empty()) {
        if (img.rows>512) {
            resize(img,img,Size(), .25, .25);
        }
    }
    return img;
}

static Mat sign(const Mat &m) {
    Mat s(m.size(),m.type(),1.0f);
    s.setTo(-1, m<0);
    return s;
}

static Mat vlad_feature(Ptr<Feature2D> f2d, const Mat &vocab, const Mat &img) {
    PROFILE
    std::vector<KeyPoint> kp;
    Mat feat, desc;
    {
        PROFILEX("vlad:detect")
        f2d->detectAndCompute(img, Mat(), kp, desc);
    }
    if (kp.size()) {
        PROFILEX("vlad:compute")
        feat = Mat(vocab.size(), CV_32F, 0.0f);
        for (int j=0; j<desc.rows; j++) {
            Mat dr = desc.row(j);
            for (int i=0; i<vocab.rows; i++) {
                Mat f = vocab.row(i) - dr;
                normalize(f,f);
                feat.row(i) += f/vocab.rows;
            }
        }
        Mat f2;
        sqrt(abs(feat), f2);
        feat = sign(feat).mul(f2);
        normalize(feat,feat);
        return feat.reshape(1,1);
    }
    return Mat();
}

static Mat bow_feature(Ptr<Feature2D> f2d, Ptr<BOWImgDescriptorExtractor> extract, const Mat &img) {
    PROFILE
    std::vector<KeyPoint> kp;
    Mat feat;
    {
        PROFILEX("bow:detect")
        f2d->detect(img, kp);
    }
    if (kp.size()) {
        PROFILEX("bow:compute")
        extract->compute(img,kp,feat);
    }
    return feat;
}

static cv::Ptr<cv::flann::Index> train_index(const Mat &trainData) {
    if (trainData.type() == CV_8U) {
        return makePtr<cv::flann::Index>(trainData, cv::flann::LinearIndexParams(), cvflann::FLANN_DIST_HAMMING);
    }
    return makePtr<cv::flann::Index>(trainData, cv::flann::LinearIndexParams(), cvflann::FLANN_DIST_L2);
}

int main( int argc, char ** argv ) {
    const char *datapath = "c:/data/faces/att";
    String fname  = argc>1 ? argv[1] : "orb";
    int nclusters = argc>2 ? atoi(argv[2]) : 512;
    int ncimages  = argc>3 ? atoi(argv[3]) : 512;

    Ptr<Feature2D> det,ext;
    Ptr<DescriptorMatcher> matcher;
    Ptr<BOWTrainer> bow_train;
    trainFeatures(fname, det, ext, matcher, bow_train, nclusters);

    Ptr<BOWImgDescriptorExtractor> bow_extract = makePtr<BOWImgDescriptorExtractor>(ext,matcher);

    vector<String> fn;
    glob(datapath, fn, true);
    cout << fn.size() << " filenames." << endl;

    Mat vocab;

    FileStorage fs(fname+".cbir.yml", 0);
    if (!fs.isOpened()) {
        int fsize=0;
        for (int i=0; i<ncimages; i++) {
            int id = theRNG().uniform(0,fn.size());
            std::vector<KeyPoint> kp;
            Mat img = getImg(fn[id], fname);
            if (img.empty()) continue;
            Mat feat;
            if (det == ext) {
                det->detectAndCompute(img, Mat(), kp, feat);
            } else {
                det->detect(img, kp);
                ext->compute(img, kp, feat);
            }
            if (feat.rows>0)
                bow_train->add(feat);
            fsize += feat.rows;
            cout << i << "\r";
            if (i%200==199)
                cout << "cluster " << fname + " " << i << " " << id << " " << feat.size() << " " << fsize << endl;
        }
        cout << bow_train->descriptorsCount() << " features." << endl;
        {
            PROFILEX("bow:cluster")
            vocab = bow_train->cluster();
        }
        FileStorage fs2(fname+".cbir.yml", 1+FileStorage::BASE64);
        fs2 << "cbir" << vocab;
        fs2.release();
    } else {
        fs["cbir"]>> vocab;
        fs.release();
    }
    cerr << "bow vocab: " << vocab.size() << endl;
    bow_extract->setVocabulary(vocab);

    int fsize=0;
    Mat trainData, indices;;
    for (int i=0; i<ncimages; i++) {
        PROFILEX("train data")
        int id = theRNG().uniform(0,fn.size());
        Mat img = getImg(fn[id],fname);
        if (img.empty()) continue;
        Mat feat = fname =="vlad" ? vlad_feature(det, vocab, img) : bow_feature(det, bow_extract, img);
        if (feat.rows) {
            trainData.push_back(feat);
            indices.push_back(id);
        }
        fsize += feat.rows;
        cout << i << "\r";
        if (i%100==99)
            cout << "train " << i << " " << id << " " << feat.size() << " " << fsize << endl;
    }
    cout << "train " << fname << " " << trainData.size() << " " << trainData.type() << endl;

    Ptr<cv::flann::Index> index;
    {
        PROFILEX("flann")
        index = train_index(trainData);
        cout << "index " << trainData.size() << " " << trainData.type() << endl;
    }

    int K=5;
    cv::flann::SearchParams params;
    for (int i=0; i<30; i++) {
        int idx = theRNG().uniform(0,indices.rows);
        int id = indices.at<int>(idx);
        Mat org = getImg(fn[id],fname);
        if (org.empty()) continue;
        cout << "test ";

        Mat feat = fname =="vlad" ? vlad_feature(det, vocab, org) : bow_feature(det, bow_extract, org);
        if (feat.empty()) continue;
        cv::Mat dists;
        cv::Mat found;
        index->knnSearch(feat, found, dists, K, params);
        Mat res;
        resize(org,res,Size(240,160));
        for (int j=1; j<K; j++) {
            int id = indices.at<int>(found.at<int>(j));
            Mat img = imread(fn[id],fname!="lucid"?0:1);
            if (img.empty()) continue;
            resize(img,img,Size(240,160));
            hconcat(res,img,res);
        }
        cout << " " << found << endl;
        imshow("I",res);
        waitKey();
    }
    return 0;
}
