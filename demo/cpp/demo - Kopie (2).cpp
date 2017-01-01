#include <set>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


void filter (const Mat &image, Mat &out)
{
    out.create(image.size(), image.type());

    for (int i=0; i<=image.rows;i++){
        for (int j=0; j<=image.cols; j++){
            Vec3b intensity = image.at<Vec3b>(i, j);
            //cerr << i << " " << j << " " << intensity << endl;
            float b   = intensity[0];
            float g   = intensity[1];
            float r   = intensity[2];
            float rg  = (r - g) * (r - g);
            float rb  = (r - b) * (r - b);
            float gb  = (g - b) * (g - b);
            float sum = (rg + rb + gb);
            float l1  = rg / sum;
            float l2  = rb / sum;
            float l3  = gb / sum;

            out.at<Vec3b>(i,j)= Vec3b(l1 * 255, l2 * 255, l3 * 255);
        }
    }
}

void filter2 (const Mat &image, Mat &out)
{
    // convert and split
    Mat imgf;
    image.convertTo(imgf, CV_32F, 1.0/255.0);
    Mat chn[3];
    split(imgf, chn);
    // (R-G)^2
    Mat rg = chn[2] - chn[1];
    multiply(rg,rg,rg);
    // (R-B)^2
    Mat rb = chn[2] - chn[0];
    multiply(rb,rb,rb);
    // (G-B)^2
    Mat gb = chn[1] - chn[0];
    multiply(gb,gb,gb);
    // divide by sum
    Mat sm = (rg + rb + gb);
    divide(rg, sm, rg);
    divide(rb, sm, rb);
    divide(gb, sm, gb);
    // merge and concvert
    Mat ch2[] = {rg, rb, gb};
    merge(ch2, 3, imgf);
    imgf.convertTo(out, CV_8U, 255.0);
}

int main(int agra, char** argv){
    String fn="phase1.png";
    if (agra>1) fn = argv[1];
    Mat image = imread(fn);
    if (image.empty())
    {
        cerr << "U$)TZ(T§$TI§H";
        return 1;
    }
    Mat result;
    filter2(image, result);
    imshow("org", image);
    imshow("res", result);
    waitKey(0);
    return 0;
}
