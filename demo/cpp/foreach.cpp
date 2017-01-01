
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include <iostream>

using namespace cv;
using namespace std;


typedef uchar Pixel;
//typedef cv::Point_<uchar> Pixel;
struct Operator {
    void operator ()(Pixel &pixel, const int * position) const
    {
        cout << format("[%d,%d]= %d \n",position[0],position[1],(int)pixel);
    }
};

int main( int argc, char* argv[])
{
    Mat mTest(Size(3, 2), CV_8UC1,Scalar(0));
    randn(mTest,Scalar(125),Scalar(125));

    cout<< format (" Size : %d , %d \n",mTest.rows,mTest.cols);

    for (int Rows = 0; Rows < mTest.rows; Rows++)
    {
        for (int Cols = 0; Cols < mTest.cols; Cols++)
        {
            cout << format("[%d,%d]= %d \t",Rows,Cols,mTest.at<uchar>(Rows,Cols));
        }
        cout << "\n";
    }
    cout << "\n\n";

    mTest.forEach<Pixel>(Operator());

    waitKey();

    return 0;
}
