#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
using namespace cv;
using namespace std;

#ifndef CV_CUBE
#  define CV_CUBE(x)  ((x)*(x)*(x))
#endif


static cv::Mat rgb2luv(const cv::Mat &src)
{
    cv::Mat dst(src.size(), src.type());

    const float a  = CV_CUBE(29.0f)/27;
    const float y0 = 8.0f/a;

    const float mX[] = {0.430574f, 0.341550f, 0.178325f};
    const float mY[] = {0.222015f, 0.706655f, 0.071330f};
    const float mZ[] = {0.020183f, 0.129553f, 0.939180f};

    const float maxi= 1.0f/270;
    const float minu=  -88*maxi;
    const float minv= -134*maxi;

    const float un = 0.197833f;
    const float vn = 0.468331f;

    // build (padded) lookup table for y->l conversion assuming y in [0,1]
    std::vector <float> lTable(1024);
    for (int i = 0; i < 1024; ++i)
    {
        float y = i/1024.0f;
        float l = y > y0 ? 116*powf(y, 1.0f/3.0f) - 16 : y*a;

        lTable[i] = l*maxi;
    }
    for (int i = 0; i < 40; ++i)
        lTable.push_back(*--lTable.end());

    const int nchannels = 3;

    for (int i = 0; i < src.rows; ++i)
    {
        const float *pSrc = src.ptr<float>(i);
        float *pDst = dst.ptr<float>(i);

        for (int j = 0; j < src.cols*nchannels; j += nchannels)
        {
            const float rgb[] = {pSrc[j + 0], pSrc[j + 1], pSrc[j + 2]};

            const float xyz[] = {mX[0]*rgb[0] + mX[1]*rgb[1] + mX[2]*rgb[2],
                                 mY[0]*rgb[0] + mY[1]*rgb[1] + mY[2]*rgb[2],
                                 mZ[0]*rgb[0] + mZ[1]*rgb[1] + mZ[2]*rgb[2]};
            const float nz = 1.0f / float(xyz[0] + 15*xyz[1] + 3*xyz[2] + 1e-35);

            const int idx = cvFloor(1024*xyz[1]);
            if (idx >= lTable.size())
                continue; //CV_Error(0,"hey");

            const float l = pDst[j] = lTable[idx];

            pDst[j + 1] = l * (13*4*xyz[0]*nz - 13*un) - minu;;
            pDst[j + 2] = l * (13*9*xyz[1]*nz - 13*vn) - minv;
        }
    }

    return dst;
}

int main() {
    Mat in = imread("c:/p/opencv/samples/data/fruits.jpg");
    Mat flt; in.convertTo(flt, CV_32F, 1.0/255);

    Mat lu1 = rgb2luv(flt);
    Mat lu2 = rgb2luv(flt);
    Mat d; absdiff(lu1,lu2,d);
    cout << sum(d) << endl;

    return 0;
}
