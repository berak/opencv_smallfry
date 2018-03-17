#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;


int main(int argc, char **argv)
{
    Mat A = imread("cilia.png");
    Mat B;
 //   logPolar(A,B,Point(A.cols/2,A.rows/2), 70, INTER_LINEAR );
    linearPolar(A,B,Point(A.cols/2,A.rows/2), A.cols/2-20, INTER_LINEAR );

    imshow("W",A);
    imshow("Q",B);
    waitKey();
}
