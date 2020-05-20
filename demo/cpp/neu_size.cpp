#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/layer.details.hpp>  // CV_DNN_REGISTER_LAYER_CLASS

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main(int argc, char** argv) {
    dnn::Net neural = dnn::readNet("c:/data/dnn/udnie.t7");
    if (neural.empty())
    {
        cout << "check net file path\n";
        return 0;
    }

    neural.setPreferableTarget(dnn::DNN_TARGET_CPU);
    neural.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    Mat src = imread("c:/p/opencv/samples/data/fruits.jpg");
    if (src.empty())
    {
        cout << "check image file path\n";
        return 0;
    }
    int pass = 1;
    resize(src, src, Size(), 0.5,0.5);
    cout << src.size() << endl;
    Mat input = dnn::blobFromImage(src, 1.0,
        src.size(),
        Scalar(103.939, 116.779, 123.68), false, false);
    Mat out;
    neural.setInput(input);
    out = neural.forward();
    pass = 2;
    Mat src2;
    resize(src, src2, Size(), 2, 2);
    input = dnn::blobFromImage(src2, 1.0,
        src2.size(),
        Scalar(103.939, 116.779, 123.68), false, false);
    neural.setInput(input);
    cout << src2.size() << endl;
    //out = neural.forward();// EXCEPTION IS HERE
    vector<Mat> outs;
    vector<string> outNames {"l17_BatchNorm", "l22_Slice"};
    neural.forward(outs,outNames);
    cout << outs[0].size << endl;
    cout << outs[1].size << endl;
    //neural.dumpToFile("style.dot");
    return 0;

}
