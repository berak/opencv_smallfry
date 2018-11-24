
#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/shape.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/superres.hpp"
#include "opencv2/video.hpp"

#include "opencv2/aruco.hpp"
#include "opencv2/bgsegm.hpp"
#include "opencv2/bioinspired.hpp"
#include "opencv2/ccalib.hpp"
#include "opencv2/dpm.hpp"
#include "opencv2/face.hpp"
#include "opencv2/line_descriptor.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/rgbd.hpp"
#include "opencv2/saliency.hpp"
#include "opencv2/stereo.hpp"
#include "opencv2/structured_light.hpp"
#include "opencv2/surface_matching.hpp"
#include "opencv2/text.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/xobjdetect.hpp"
#include "opencv2/xphoto.hpp"

using namespace cv;
#include <algorithm>
#include <iostream>
#include <chrono> // c++11 !!
#include <numeric>
#include <vector>
#include <bitset>
#include <map>
#include <set>
using namespace std;

void download(const char * url, const char * localthing) {
    int n = system(format("curl -s -o %s '%s'",localthing,url).c_str());
}
Mat urlimg(const char * url) {
    download(url,"local.img");
    Mat im = imread("local.img", -1);
    return im;
}
int main()
{
    Mat ocv = imread("input.img",-1);
    if ( ocv.empty() )
        ocv = Mat(8,8,CV_8UC3,Scalar(40,40,40));

;;
    imwrite("output.png", ocv);
    return 0;
}
