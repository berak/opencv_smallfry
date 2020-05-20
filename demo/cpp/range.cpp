#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main(int argc, char** argv) {
    std::ifstream in("range.txt");
    Mat_<float> data, angle;
    float tok,a=0;
    while(in >> tok) {
        data.push_back(tok);
        angle.push_back(a++);
    }
    in.close();

    Mat_<float> x,y;
    polarToCart(data, angle, x, y, true);

    float thresh = 255.0f;
    int h = 620, w = 620;
    Point2f center(w/2, h/2);
    Mat draw(h,w,CV_8UC3,Scalar::all(128));
    circle(draw, center, thresh, Scalar(80,80,80),-1);
    circle(draw, center, 5, Scalar(0,120,0),-1);

    for (int r=0; r<data.rows; r++) {
        Point2f p(x(r), y(r)),
                q(p + center); // for drawing
        float dist = norm(q - center);
        float col = (dist < thresh) ? 230 : 150;
        circle(draw, q, 4, Scalar(50,col,50),2);
    }

    imshow("F",draw);
    waitKey();
    return 0;
}
