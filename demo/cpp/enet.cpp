#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace std;
int main(int argc, char** argv) {
	string nn = argc>1 ? argv[1] :  "c:/data/dnn/problem/ENet2.t7";
	string nt = argc>2 ? argv[2] :  "";
    dnn::Net net;
    net = dnn::readNet(nn,nt);
    return 0;
}