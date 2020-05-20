#include <iostream>

#include <opencv2/core.hpp>
using namespace cv;
using namespace std;
int main(int argc, char** argv) {
    cv::FileStorage fs1 = cv::FileStorage("cal.yml", cv::FileStorage::READ);
    cv::FileStorage fs = fs1;
    fs1.release();

    cout << "open " <<fs.isOpened() << endl;

    cv::FileNode node = fs.root();
    //cv::FileNode top = fs.getFirstTopLevelNode();
    //std::cout << "Got top node: " << fs.elname << std::endl;
    std::cout << "root.empty() " << node.empty() << std::endl;
    std::cout << "Got root node: " << node.type() << std::endl;
    //std::cout << "Root node is type: " << node.type() << std::endl; //Crash here
    int ad; fs["board_height"] >> ad;
    cout << "ad " << ad << endl;

    return 0;
}