//#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

class point3D_t : public cv::Point3f { };
class point2D_t : public cv::Point2f { };

int main()
{

  std::vector<point3D_t> input;
  std::vector<point2D_t> output;

  cv::Mat R,T,K;
  std::vector<double> distCoeffs;

  cv::projectPoints(input, R, T, K, distCoeffs, output);	return 0;
  return 0;

}
