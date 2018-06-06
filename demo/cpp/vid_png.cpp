#include "opencv2/opencv.hpp"
#include <fstream>

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
    VideoCapture cap("logo.png");
    cout << cap.isOpened() << endl;
}
