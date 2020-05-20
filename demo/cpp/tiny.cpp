#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
int main(int argc, char** argv) {
	string path = "c:/data/dnn/";
    dnn::Net net = dnn::readNet(path+"tiny.weights", path+"yolov3-tiny.cfg");
    for (string s:net.getLayerNames()) {
    	cout << s << endl;
    }
    Mat img = imread("box.png");
    Mat blob = dnn::blobFromImage(img, 0.00392, Size(416, 416), Scalar(0), true, false);
    net.setInput(blob);
    net.dumpToFile("tiny.dot");
    std::vector<String> outn {"yolo_16", "yolo_23"};
	std::vector<Mat> outs;
    net.forward(outs, outn);
    return 0;
}