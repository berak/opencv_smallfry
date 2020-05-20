#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;


using namespace cv;
using namespace std;
int main(int argc, char** argv) {
    string folder = "c:/data/dnn/tcnn/";

    dnn::Net net = dnn::readNet(folder + "vanilla_deplay.prototxt", folder + "vanillaCNN.caffemodel");

    string imp = "C:\\data\\faces\\lfw40_crop\\Abdullah_Gul_0002.jpg";
    Mat img = imread(imp);
    cout << img.size() << endl;

    auto run = [&](const Mat &blob) {
        net.setInput(blob);
        Mat out = net.forward();
        cout << out.size << endl;
        cout << out << endl;
    };

    run(dnn::blobFromImage(img,1,Size(40,40)));

    vector<Mat> vec(10,img);
    run(dnn::blobFromImages(vec,1,Size(40,40)));

    vector<Mat> pred;

    net.setInput(dnn::blobFromImages(vec,1,Size(40,40)));
    net.forward(pred);
    cout << pred.size() << endl;
    cout << pred[0] << endl;

    //cout << pred[0][0] << endl;

    return 0;
}