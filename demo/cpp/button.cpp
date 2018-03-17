#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;

void pick_button(int e, int x, int y, int s, void *param)
{
    Rect r = *((Rect*)param);
    if (e==1) // lbutton down
    {
        if (r.contains(Point(x,y)))
            cerr << "HIT!" << endl;
    }
}

int main( int argc, char** argv )
{
    Rect r(50,50,50,50);

    namedWindow("blue");
    setMouseCallback("blue", pick_button, (void*)(&r));
    Mat draw(400,400,CV_8UC3, Scalar::all(127));
    rectangle(draw, r, Scalar(200,0,0),-1);
    imshow("blue", draw);
    waitKey();
    return 0;
}
