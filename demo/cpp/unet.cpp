#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace std;
int main(int argc, char** argv) {
    dnn::Net net;
    net = dnn::readNet("c:/data/dnn/problem/problem.onnx");
    return 0;
}