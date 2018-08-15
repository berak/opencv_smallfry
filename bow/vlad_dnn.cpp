#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/core/utils/filesystem.hpp>
#include "opencv2/dnn.hpp"

#include "profile.h"

#include <cstdio>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

int missedImg=0;
int missedFeat=0;


static Mat sign(const Mat &m) {
    PROFILE
    Mat s(m.size(),m.type(),1.0f);
    s.setTo(-1, m<0);
    return s;
}

static Mat vlad_feature(Mat feats, const Mat &vocab) {
    PROFILE
    Mat vlad = Mat(vocab.size(), CV_32F, 0.0f);
    for (int i=0; i<feats.rows; i++) {
        Mat f = feats.row(i);
        double bd = 9999999;
        int bi = -1;
        for (int j=0; j<vocab.rows; j++) {
            double d = norm(f,vocab.row(j));
            if (d < bd) {
                bd = d;
                bi = j;
            }
        }
        Mat res = f - vocab.row(bi);
        normalize(res, res); // innorm
        vlad.row(bi) += res / vocab.rows;
    }
    // power normalization
    Mat f2;
    sqrt(abs(vlad), f2);
    vlad = sign(vlad).mul(f2);
    // L2 normalization
    normalize(vlad, vlad);
    return vlad.reshape(1,1);
}

static cv::Ptr<cv::flann::Index> train_index(const Mat &trainData) {
    PROFILE
    if (trainData.type() == CV_8U) {
        return makePtr<cv::flann::Index>(trainData, cv::flann::LinearIndexParams(), cvflann::FLANN_DIST_HAMMING);
    }
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
    //const char *datapath = "c:/data/faces/att";
    String fname  = argc>1 ? argv[1] : "vlad_dnn";
    int nclusters = argc>2 ? atoi(argv[2]) : 64;
    int ncimages  = argc>3 ? atoi(argv[3]) : 2000; // can't handle more
    int ntimages  = argc>4 ? atoi(argv[4]) : 100;
    //theRNG().state = getTickCount();
    ocl::setUseOpenCL(false);
   // dnn::Net net = dnn::readNetFromTensorflow("c:/data/mdl/tensorflow_inception_graph.pb");
    std::string modelTxt = "c:/data/mdl/squeezenet/deploy.prototxt";
    std::string modelBin = "c:/data/mdl/squeezenet/squeezenet_v1.1.caffemodel";
    dnn::Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
    cv::Size inputImgSize = cv::Size(227, 227);

    auto dnn_feature = [&](const Mat &img) {
        net.setInput(dnn::blobFromImage(img, 1, inputImgSize, Scalar(), false));
        Mat blob = net.forward("fire9/squeeze1x1"); // [1, 512, 14, 14]
        return blob.reshape(1,blob.size[1]);
    };

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


    Mat vocab;
    // 2. build voc
    FileStorage fs1(fname+".voc.yml", 0);
    if (!fs1.isOpened()) {
        Mat vocfeat;
        for (int i=0; i<ncimages; i++) {
            int id = theRNG().uniform(ntimages, fn.size()-ntimages);
            Mat img = imread(fn[id]);
            Mat feat = dnn_feature(img);
            vocfeat.push_back(feat);
            cout << i << "\r";
            if (i%100==99)
                cout << "feature " << i << vocfeat.size() << endl;
        }
        Mat labels;
        cout << "starting to cluster " << vocfeat.size() << " descriptors." << endl;
        cv::kmeans(vocfeat, nclusters, labels, cv::TermCriteria(), 3, cv::KMEANS_PP_CENTERS, vocab);
        vocfeat.release();

        FileStorage fs2(fname+".voc.yml", 1+FileStorage::BASE64);
        fs2 << "voc" << vocab;
        fs2.release();
    } else {
        fs1["voc"] >> vocab;
        fs1.release();
    }
    cout << "vocab " << vocab.size() << endl;
    Mat trainData, testData;
    FileStorage fs3(fname+".train.yml", 0);
    if (!fs3.isOpened()) {
        for (int i=0; i<fn.size(); i++) {
            PROFILEX("data")
            //String cat = category(fn[i],datapath.size()+1);
            Mat img = imread(fn[i]);
            Mat f = dnn_feature(img);
            //cout <<f.size() << endl;
            Mat feat = vlad_feature(f, vocab);
            if (feat.rows) {
                if (i < ntimages)
                    testData.push_back(feat);
                else
                    trainData.push_back(feat);
            } else {
                missedFeat ++;
            }
            cout << i << "\r";
            if (i%100==99)
                cout << "train " << i << " " << testData.size() << " " << trainData.size() << endl;
        }

        FileStorage fs2(fname+".train.yml", 1+FileStorage::BASE64);
        fs2 << "train" << trainData;
        fs2 << "test" << testData;
        fs2.release();
    } else {
        fs3["train"] >> trainData;
        fs3["test"] >> testData;
        fs3.release();
    }
    cout << "train " << fname << " " << trainData.size() << " " << trainData.type() <<  " " << missedFeat << endl;

    // 3. build the index
    Ptr<cv::flann::Index> index = train_index(trainData);
    cout << "index " << trainData.size() << " " << trainData.type() << endl;

    // 4. run tests
    int K=5;
    float correct=0; int ntests=0;
    cv::flann::SearchParams params;
    for (int i=0; i<testData.rows; i++) {
        String cat = category(fn[i], datapath.size() + 1) ;
        if (cat == "faces_easy") cat = "faces";

        Mat feat = testData.row(i);
        if (feat.empty()) continue;
        cv::Mat dists;
        cv::Mat found;
        index->knnSearch(feat, found, dists, K, params);
        cout << "test " << cat << " " << i << " " << found << " " << dists << endl;

        Mat res = imread(fn[i]);
        resize(res,res,Size(240,160));
        for (int j=0; j<K; j++) {
            int f = found.at<int>(0,j);
            Mat img = imread(fn[f]);
            if (img.empty()) continue;
            String cat2 = category(fn[f], datapath.size() + 1) ;
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
