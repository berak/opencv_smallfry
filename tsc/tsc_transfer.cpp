#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#include "printnet.h"

int WINSIZE = 224;

//
// transfer learning using inceptionv5 as a fixed preprocessing pipeline,
//   and opencv's ANN_MLP at the top
//

//! load images for max_classes classes from train or test dir.
//  note, that the csv files *claim* to have W,h,x,y order, but actually, it's the other way round ! (H,W,y,x)
double cv_load(const String &dir, vector<Mat> &data, Mat &labels, int max_classes=-1, bool gray=true, int skip=0)
{
    data.clear();
    labels.release();
    int64 t0 = getTickCount();
    int k = 0;
    vector<String> csvs;
    glob(dir + "*.csv", csvs, true);
    for (auto cn : csvs) {
        ifstream csv(cn);
        string file;
        getline(csv, file); // skip csv header
        while(csv.good()) {
            k++;
            char c = 0;
            file = "";
            // avoid "evil string globbing"
            while (csv.good()) {
                csv.get(c);
                if (c == ';') break;
                if (c == '\n') continue;
                file += c;
            }
            if (file.empty()) break;

            int W,H,x1,y1,x2,y2,label;
            csv >> H >> c >> W >> c >> y1 >> c >> x1 >> c >> y2 >> c >> x2 >> c >> label;
            if ((skip > 1) && (k % skip != 0)) continue;
            if ((max_classes > 0) && (label >= max_classes)) break;

            String fn = dir + format("%05d/",label) + file;
            Mat img = imread(fn);
            if (img.empty()) continue;

            cv::Mat resized;
            Rect roi(Point(x1,y1), Point(x2,y2));
            cv::resize(img(roi), resized, cv::Size(WINSIZE, WINSIZE));
            data.push_back(resized);
            labels.push_back(label);
        }
    }
    cout << "loaded " << data.size() << " images" << endl;
    int64 t1 = getTickCount();
    return  ((t1-t0)/getTickFrequency());
}

void cv_results(int classes, Mat &results, Mat &labels, const String &title)
{
    CV_Assert(results.rows==labels.rows && results.cols==1 && labels.cols==1);

    Mat_<int> confusion(classes, classes, 0);
    for (int i=0; i<results.rows; i++) {
        int p = (int)results.at<float>(i);
        int t = (int)labels.at<int>(i);
        confusion(p, t) ++;
    }
    const int MAXC = 30; // try to fit console win
    if (confusion.rows < MAXC)
        cout << title << " confusion:\n" << confusion << endl;
    else // skip results beyond MAXC
        cout << title << format(" confusion (showing %d of %d):\n", MAXC, confusion.rows) << confusion(Rect(0, 0, MAXC, MAXC)) << endl;

    // accuracy:
    float correct  = sum(confusion.diag())[0];
    float accuracy = correct / results.rows;
    cout << title << " accuracy: " << accuracy << endl;
}

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | show this message }"
        "{ batch b        |    16| batch size for dnn training }"
        "{ maxc M         |    62| restrict to M classes (for speedup, originally, there are 62) }"
        "{ l1             |   256| first layer }"
        "{ l2             |   128| 2nd layer }"
        "{ learn l        |0.0004| initial learning rate for the (newly added) ANN layers }"
        "{ input i        |input | how to find the input in the model file }"
        "{ output o       |head1_bottleneck/reshape| stop forward pass here }"
        "{ data D         |C:/data/BelgiumTSC/| path to dataset }"
        "{ model m        |c:/data/dnn/tensorflow_inception_graph.pb| path to model }"
        "{ proto p        |      | proto file }"
        "{ print P        |      | print out net structure }"
    );

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    string tscdir = parser.get<string>("data");
    int max_classes = parser.get<int>("maxc");
    int batch_size = parser.get<int>("batch");
    int l1 = parser.get<int>("l1");
    int l2 = parser.get<int>("l2");
    float learn = parser.get<float>("learn");

    string inBlobName  = parser.get<string>("input");
    string outBlobName = parser.get<string>("output");
    string modelFile   = parser.get<string>("model");
    string protoFile   = parser.get<string>("proto");
    dnn::Net net = (protoFile=="") ?
        dnn::readNetFromTensorflow(modelFile):
        dnn::readNetFromCaffe(protoFile, modelFile);
    CV_Assert(!net.empty());

    if (parser.has("print")) printnet(net, batch_size, 3, WINSIZE, WINSIZE);

    Mat labels, features;
    vector<Mat> data;
    auto transfer = [&]() { // reused for the test batches
        for (int i=0; i<int(data.size())-batch_size; i+=batch_size) {
            vector<Mat> batch(data.begin() + i, data.begin() + i + batch_size);
            Mat blob = dnn::blobFromImages(batch);  // Convert Mat to image batch
            blob -= 117.0;                          // magic mean
            net.setInput(blob, inBlobName);
            Mat result = net.forward(outBlobName);  // sadly, batches are not any faster
            result = result.reshape(1,batch_size);
            features.push_back(result);
            cout << " [" << blob.size[0] << " " << blob.size[1] << " " << blob.size[2] << " " << blob.size[3] << "]";
            cout << " [" << result.size[0] << " " << result.size[1] << "] "<< features.size() <<"\r";
        }
        cout << "\n";
        labels.resize(features.rows); // since i'm throwing away features
    };

    FileStorage fs("trainf.yml.gz",0);
    if (! fs.isOpened()) {
        double t = cv_load(tscdir + "Training/", data, labels, max_classes);
        transfer();

        FileStorage fs1("trainf.yml.gz",1);
        fs1 << "features" << features;
        fs1 << "labels" << labels;
        fs1.release();
    } else {
        fs["features"] >> features;
        fs["labels"] >> labels;
        fs.release();
    }

    Mat_<int> layers(4, 1);
    layers << features.cols, l1, l2, max_classes;
    cout << "nn " << layers.t() << endl;
    Ptr<ml::ANN_MLP> nn = ml::ANN_MLP::create();
    nn->setLayerSizes(layers);
    nn->setTrainMethod(ml::ANN_MLP::BACKPROP, learn);
    nn->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
    nn->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 300, 0.0001));

    // mlp needs "one-hot" encoded responses for training
    Mat hot(labels.rows, max_classes, CV_32F, 0.0f);
    for (int i=0; i<labels.rows; i++)
    {
        int id = (int)labels.at<int>(i);
        hot.at<float>(i, id) = 1.0f;
    }

    nn->train(features, 0, hot);

    data.clear();
    features.release();
    labels.release();
    FileStorage fs2("testf.yml.gz",0);
    if (! fs2.isOpened()) {
        double t2 = cv_load(tscdir + "testing/", data, labels, max_classes);
        transfer();
        FileStorage fs3("testf.yml.gz",1);
        fs3 << "features" << features;
        fs3 << "labels" << labels;
        fs3.release();
    } else {
        fs2["features"] >> features;
        fs2["labels"] >> labels;
        fs2.release();
    }

    Mat results;
    for (int r=0; r<features.rows; r++) {
        float p = nn->predict(features.row(r));
        results.push_back(p);
    }

    cv_results(max_classes, results, labels, "transfer");
    return 0;
}
