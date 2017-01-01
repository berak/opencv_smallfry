
#include <opencv2/opencv.hpp>
#include <iostream>
#include <deque>

using namespace cv;
using namespace std;


int main()
{
    VideoCapture cap(0); // open the default camera
    if (!cap.isOpened())  // check if we succeeded
        return -1;

    std::deque<cv::Mat> queue;
    while(true)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break; // end of video stream
        queue.push_back(frame);
        if (queue.size() > 15) {
            queue.pop_front();
            Mat avgImg(frame.size(), CV_32FC3, Scalar());
            for (int i=0; i<15; i++)
                accumulate(queue[i], avgImg);
            avgImg.convertTo(avgImg, CV_8UC3, 1.0/15);
            imshow("avg", avgImg);
        }
        imshow("webcam", frame);
        if (waitKey(10) > 0) break;
    }
    return EXIT_SUCCESS;
}
