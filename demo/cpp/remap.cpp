#include "opencv2/opencv.hpp"
#include "opencv2/tracking.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    const float a = 0.27; // it's a constant ?
    Mat img = imread("../img/grid.jpg");
    Mat_<Vec2f> prjMap(img.size());
    for (int x=0; x<img.cols; x++) {
        for (int y=0; y<img.rows; y++) {
            float u = x - y * tan(a);
            float v = y;
            prjMap(y, x) = Vec2f(u,v);
        }
    }
    Mat projected;
    remap(img, projected, prjMap, cv::Mat(), INTER_LINEAR);
    imshow("P", projected);
    waitKey();
    return 0;
}
