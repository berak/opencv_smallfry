#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace std;
int main(int argc, char** argv) {
    Mat m(240,240,CV_8UC3);
    Mat b; dnn::blobFromImage(m,b);
    cout << b.size << endl;

    string folder = "c:/data/dnn/";
    dnn::Net net;
    //net = dnn::readNet(folder+"enetb0-coco_final.weights", folder+"efficientnet_b0.cfg");
    //net = dnn::readNet(folder+"efficientnet_b0.onnx");
    net = dnn::readNet(folder+"sphereface_model.caffemodel", folder+"sphereface_model.prototxt");
    net.setInput(b);
    vector<string> names = net.getLayerNames();
    for (auto n:names)
    	cout << n << endl;
    net.dump();
    //net.dumpToFile("eff.txt");
    return 0;
}