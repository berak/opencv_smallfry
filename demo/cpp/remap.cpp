#include "opencv2/opencv.hpp"
#include "opencv2/tracking.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    Mat img = imread("img/grid.jpg");
    int width  = img.cols;
    int height = img.rows;
    Mat_<Vec2f> prjMap(height, width);
    for (int x=0; x<width; x++) {
        for (int y=0; y<height; y++) {
            float u = x + sin(float(y)/100)*15;
            float v = y + cos(float(x)/100)*20;
            prjMap(y, x) = Vec2f(u,v);
        }
    }
    Mat projected;
    remap(img, projected, prjMap, cv::Mat(), INTER_LINEAR);
    imshow("P", projected);
    waitKey();
    return 0;
}
