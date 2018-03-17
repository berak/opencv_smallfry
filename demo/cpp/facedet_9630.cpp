#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>

using namespace cv;
using namespace std;

extern const uint8_t img[];
extern const uint8_t tmpl[];

int main(int argc, const char* argv[]) {
//ocl::setUseOpenCL(0);
Mat image = imread("img/h4.jpg",0);
CascadeClassifier faceCascade("haarcascade_frontalface_alt.xml");
vector<Rect> facesM, facesU;

faceCascade.detectMultiScale(image, facesM);
cout << "Mat" << endl << Mat(facesM) << endl << endl;

UMat imageUMat = image.getUMat(cv::ACCESS_READ);
faceCascade.detectMultiScale(imageUMat, facesU);
cout << "UMat" << endl << Mat(facesU) << endl << endl;
cout << cv::getBuildInformation();
  return 0;
}

