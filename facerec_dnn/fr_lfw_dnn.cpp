/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2014, Itseez Inc, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Itseez Inc or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/datasets/fr_lfw.hpp"

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::datasets;


class FaceNet
{
    dnn::Net net;
    Ptr<ml::SVM> svm;
    Mat svm_data;
    Mat svm_labels;
    double _threshold;
    map<String,Mat> cache;
public:
    const int FIXED_FACE = 96;

    FaceNet(const String &pathToTorchnet)
        : svm(ml::SVM::create())
    {
        svm->setKernel(ml::SVM::LINEAR);
        //svm->setC(100);
        try {
            net = dnn::readNetFromTorch(pathToTorchnet);
        } catch(Exception &e) {
            cerr << "Download it from:  ";
            cerr << "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7" << endl;
        }
    }

    int size() { return cache.size(); }

    Mat process(const String &imagepath)
    {
        Mat f = cache[imagepath];
        if (! f.empty()) {
            return f;
        }
        Mat image = imread(imagepath);
        CV_Assert(!image.empty());
        Mat roi = image(Rect(77,77,FIXED_FACE,FIXED_FACE));
        Mat inputBlob = dnn::blobFromImage(roi, 1./255, Size(FIXED_FACE, FIXED_FACE), Scalar(), true, false);
        net.setInput(inputBlob);
        Mat f2 = net.forward().clone();
        cache[imagepath] = f2;
        //imshow("roi",roi);
        //waitKey(10);
        return f2;
    }

    double distance(const String &img1, const String &img2)
    {
        Mat a = process(img1);
        Mat b = process(img2);
        double d = norm(a,b);
        return d;
    }

    Mat svm_proc(const String &img1, const String &img2)
    {
        Mat a=process(img1);
        Mat b=process(img2);
        Mat c;
        //absdiff(a,b,c);
        //hconcat(a,b,c);
        multiply(a,b,c);
        //normalize(c,c);
        return c;
    }
    void svm_add(const String &img1, const String &img2, bool  same)
    {
        Mat a=svm_proc(img1,img2);
        svm_data.push_back(a);
        svm_labels.push_back(same?1:-1);
    }
    bool svm_predict(const String &img1, const String &img2)
    {
        Mat a=svm_proc(img1,img2);
        //cout << endl << "svm_predict " << a.size() << a.type() << endl << endl;

        float p = svm->predict(a);
        return p>0.01f;
    }
    bool svm_train()
    {
        //cout << endl << "svm_train " << svm_data.size() << svm_data.type() << endl << endl;
        return svm->train(svm_data,0,svm_labels);
    }

    void write(FileStorage& sav) const
    {
        sav << "size" << int(cache.size());
        sav << "keys" << "[";
        for (map<String,Mat>::const_iterator it = cache.begin(); it!=cache.end(); ++it) {
            sav << "{" << "k" << it->first << "v" << it->second << "}";
        }
        sav << "]";
    }
    void read(const FileNode& fs)
    {
        FileNode n = fs["keys"];
        FileNodeIterator it = n.begin(), it_end = n.end();
        for (; it != it_end; ++it) {
            String k; (*it)["k"] >> k;
            Mat v; (*it)["v"] >> v;
            cache[k]=v;
        }
    }
};

int main(int argc, const char *argv[])
{
    const char *keys =
            "{ help h usage ? |    | show this message }"
            "{ skip k         |1| skip value from dataset }"
            "{ scale s        |1.15| scale threshold }"
            "{ svm S          |false| use svm classifier }"
            "{ facenet f      || (required) path to facenet model }"
            "{ path p         || (required) path to dataset (lfw2 folder) }"
            "{ thresh T       |0| test only with fixed threshold }"
            "{ train t        |dev | train method: 'dev'(pairsDevTrain.txt) or 'split'(pairs.txt) }";

    CommandLineParser parser(argc, argv, keys);
    const string path(parser.get<string>("path"));
    const bool useSVM(parser.get<bool>("svm"));
    double threshold = 0;
    const double fac= parser.get<double>("scale");
    const double testThreshold=parser.get<double>("thresh");
    string trainMethod(parser.get<string>("train"));
    if (parser.has("help") || path.empty())
    {
        parser.printMessage();
        return -1;
    }

    FaceNet facenet(parser.get<string>("facenet"));
    FileStorage cc("faces.xml.gz",0);
    if (cc.isOpened())
        facenet.read(cc.root());
    printf("cache size %u\n", facenet.size());

    // load dataset
    Ptr<FR_lfw> dataset = FR_lfw::create();
    dataset->load(path);
    int steps = parser.get<int>("skip");
    unsigned int numSplits = dataset->getNumSplits();
    printf("splits number: %u %s\n", numSplits, trainMethod.c_str());
    if (trainMethod == "dev")
        printf("train size: %u\n", (unsigned int)dataset->getTrain().size());
    else
        printf("train size: %u\n", (numSplits-1) * (unsigned int)dataset->getTest().size());
    printf("test size: %u\n", (unsigned int)dataset->getTest().size());

    if (trainMethod == "dev" && testThreshold==0) // train on personsDevTrain.txt
    {
        // collect average same-distances:
        double avg = 0;
        int count = 0;
        for (unsigned int i=0; i<dataset->getTrain().size(); i+=steps)
        {
            FR_lfwObj *example = static_cast<FR_lfwObj *>(dataset->getTrain()[i].get());
            if (useSVM) {
                facenet.svm_add(path+example->image1, path+example->image2, example->same);
            } else {
                if (example->same)
                {
                    double dist = facenet.distance(path+example->image1, path+example->image2);
                    avg += dist;
                    count ++;
                    cout << format("...  %6d %6d %3.5f %3.5f  \r", count, facenet.size(), dist, avg/count);
                }
            }
        }

        if (useSVM) {
            facenet.svm_train();
        } else {
            threshold = fac * avg / count;
        }
    }
    Mat_<int> confusion = Mat_<int>::zeros(2,2);
    vector<double> p;
    for (unsigned int j=0; j<numSplits; ++j)
    {
        if (trainMethod == "split" && testThreshold==0) // train on the remaining 9 splits from pairs.txt
        {
            double avg = 0;
            int count = 0;
            for (unsigned int j2=0; j2<numSplits; ++j2)
            {
                if (j==j2) continue; // skip test split for training

                vector < Ptr<Object> > &curr = dataset->getTest(j2);
                for (unsigned int i=0; i<curr.size(); i+=steps)
                {
                    FR_lfwObj *example = static_cast<FR_lfwObj *>(curr[i].get());
                    if (useSVM) {
                        facenet.svm_add(path+example->image1, path+example->image2, example->same);
                    } else {
                        if (example->same)
                        {
                            double dist = facenet.distance(path+example->image1, path+example->image2);
                            avg += dist;
                            count ++;
                            cout << format("...  %6d %6d %3.5f %3.5f  \r", count, facenet.size(), dist, avg/count);
                        }
                    }
                }
            }
            if (useSVM) {
                facenet.svm_train();
            } else {
                threshold = fac * avg / count;
            }
        }
        if (testThreshold != 0)
            threshold = testThreshold;
        int count = 0;
        double md = 0;
        unsigned int incorrect = 0, correct = 0;
        vector < Ptr<Object> > &curr = dataset->getTest(j);
        for (unsigned int i=0; i<curr.size(); i+=steps)
        {
            FR_lfwObj *example = static_cast<FR_lfwObj *>(curr[i].get());
            bool same;
            if (useSVM) {
                same = facenet.svm_predict(path+example->image1, path+example->image2);
            } else {
                double dist = facenet.distance(path+example->image1, path+example->image2);
                same = (dist < threshold);
                md += dist;
            }
            if (same == example->same)
                correct++;
            else
                incorrect++;
            count += 1;
            cout << (example->same ? "+++" : "---");
            cout << format("  %5d %5d %5d %d %3.4f %3.5f \r", i, facenet.size(), correct, incorrect, (double(md)/count), 1.0-(double(incorrect)/(correct+1)));
            confusion(example->same, same) ++;
        }
        p.push_back(1.0*correct/(correct+incorrect));
        printf("correct: %u of %u -> %f acc, %3.5f thresh, %3.5f mean\n", correct, correct+incorrect, p.back(), threshold, (md/count));
    }
    cout << confusion << endl;
    FileStorage sav("faces.xml.gz",1+FileStorage::BASE64);
    facenet.write(sav);
    sav.release();

    double mu = 0.0;
    for (vector<double>::iterator it=p.begin(); it!=p.end(); ++it)
    {
        mu += *it;
    }
    mu /= p.size();
    double sigma = 0.0;
    for (vector<double>::iterator it=p.begin(); it!=p.end(); ++it)
    {
        sigma += (*it - mu)*(*it - mu);
    }
    sigma = sqrt(sigma/p.size());
    double se = sigma/sqrt(double(p.size()));
    printf("estimated mean accuracy: %f and the standard error of the mean: %f\n", mu, se);

    return 0;
}
