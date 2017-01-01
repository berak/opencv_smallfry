#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <bitset>

typedef cv::Mat Image;

int main(int argc, const char * argv[])
{
 uint64 t0 = cv::getTickCount();
    cv::Mat imgC = cv::imread("key1.jpg");
    cv::Mat img;
    cv::cvtColor(imgC,img,CV_BGR2GRAY);

    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    //detector->detect(img, keypoints);

    //if(keypoints.size()>10) {
        detector->detectAndCompute(img,cv::Mat(),keypoints, descriptor);
    //}

    std::cout << "Descriptor (" << descriptor.rows << "," <<
                descriptor.cols << ")" << std::endl;

    std::vector<std::bitset<256>> descriptors;
 uint64 t1 = cv::getTickCount();
    for (unsigned i=0, j=descriptor.rows; i<j; ++i)
    {
        std::bitset<256> tmp;
        int pos = 0;
        for ( unsigned l=0, k=descriptor.cols; l<k; ++l )
        {
            std::bitset<8> p = descriptor.at<uchar>(i,l);
            for(int k=0; k<8; ++k)
            {
                tmp[pos] = p[k];
                ++pos;
            }
        }
        descriptors.emplace_back(tmp);
    }
  uint64 t2 = cv::getTickCount();

    // first descriptor
    std::cout << descriptors[0] << std::endl;
 	std::cout << (t1-t0)/cv::getTickFrequency() << std::endl;
 	std::cout << (t2-t1)/cv::getTickFrequency() << std::endl;

    return 0;
}
