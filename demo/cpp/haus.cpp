#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int distance(vector<Point>const& image, vector<Point>const& tempImage)
{
    int maxDistance = 0;

    for(Point imagePoint: image)
    {
        int minDistance = numeric_limits<int>::max();

        for(Point tempPoint: tempImage)
        {
            Point diff = imagePoint - tempPoint;
            int length = (diff.x * diff.x) + (diff.y * diff.y);

            if(length < minDistance) minDistance = length;
            if(length == 0) break;
        }
        maxDistance += minDistance;
    }
    return maxDistance;
}

double hausdorffDistance(vector<Point>const& image, vector<Point>const& tempImage)
{
    double maxDistImage = distance(image, tempImage);
    double maxDistTemp = distance(tempImage, image);

    return sqrt(max(maxDistImage, maxDistTemp));
}

vector<Mat> buildPyramids(Mat& frame)
{
    vector<Mat> pyramids;

    int count = 6;

    Mat prevFrame = frame, nextFrame;

    while(count > 0)
    {
        resize(prevFrame, nextFrame, Size(), .85, .85);
        prevFrame = nextFrame;

        pyramids.push_back(nextFrame);

        --count;
    }

    return pyramids;
}

vector<Rect> slidingWindows(Mat& image, int stepSize, int width, int height)
{
    vector<Rect> windows;

    for(size_t row = 0; row < image.rows; row += stepSize)
    {
        if((row + height) > image.rows) break;

        for(size_t col = 0; col < image.cols; col += stepSize)
        {
            if((col + width) > image.cols) break;

            windows.push_back(Rect(col, row, width, height));
        }
    }

    return windows;
}

int main(int argc, const char* argv[]) {
	int T = 200;
    Mat img = imread("scene.png",0) > T;
    Mat tmp = imread("templ.png",0) > T;
    imshow("I",img);
    imshow("T",tmp);
    waitKey();
  	return 0;
}

