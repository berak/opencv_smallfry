#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/bioinspired.hpp>
#include <fstream>

using namespace cv;

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat frame = imread("pers2.jpg");
    Mat saliencyMap, binaryMap;
    Ptr<saliency::Saliency> saliencyAlgorithm = saliency::StaticSaliencyFineGrained::create();
    if( saliencyAlgorithm->computeSaliency( frame, saliencyMap ) )
    {
    //  saliency::StaticSaliencySpectralResidual spec;
    //  spec.computeBinaryMap( saliencyMap, binaryMap );
    }

    imshow("M",frame);
    //imshow("B",binaryMap);
   	imshow("S",saliencyMap);

    Mat parvo,magno;
    cv::Ptr<cv::bioinspired::Retina> myRetina = cv::bioinspired::Retina::create(frame.size());
    myRetina->run(frame);
    myRetina->getParvo(parvo);
    myRetina->getMagno(magno);
    cv::imshow("Retina Parvo", parvo);
    cv::imshow("Retina Magno", magno);


  	waitKey();
    return 0;
}
