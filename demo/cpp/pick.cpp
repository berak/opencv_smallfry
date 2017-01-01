#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;

Mat im_hsv, dist;
void pick_color(int e, int x, int y, int s, void *)
{
    if (e==1)
    {
        int r = 3;
        int off[9*2] = {0,0, -r,-r, -r,0, -r,r, 0,r, r,r, r,0, r,-r, 0,-r};
        for (int i=0; i<9; i++)
        {
            Vec3b p = im_hsv.at<Vec3b>(y+off[2*i], x+off[2*i+1]);
            cerr << int(p[0]) << " " << int(p[1]) << " " << int(p[2]) << endl;
            dist.push_back(p);
        }
    }
}

int main( int argc, char** argv )
{
    namedWindow("blue");
    setMouseCallback("blue", pick_color);

    String c_in = "."; // check a whole folder.
    if (argc>1) c_in = argv[1]; // or an image
    vector<String> fn;
    glob(c_in, fn, true);
    for(size_t i=0; i<fn.size(); i++)
    {
        Mat im_bgr = imread(fn[i]);
        if (im_bgr.empty()) continue;
        cvtColor(im_bgr, im_hsv, COLOR_BGR2HSV);
        imshow("blue", im_bgr);
        int k = waitKey() & 0xff;
        if (k==27) break; // esc.
    }
    Scalar m,v;
    meanStdDev(dist,m,v);
    cerr << "mean, var: " << endl;
    cerr << m[0] << " " << m[1] << " " << m[2] << " " << v[0] << " " << v[1] << " " << v[2] << endl;
    return 0;
}
