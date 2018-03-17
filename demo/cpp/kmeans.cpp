#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>

using namespace cv;
using namespace std;



int main(int argc, const char* argv[]) {
Mat ocv = imread("../img/exampleleft_s.jpg");
resize(ocv,ocv,Size(640,480));
Mat data;
ocv.convertTo(data,CV_32F);
data = data.reshape(1,data.total());
cerr << data.type() << " " << data.size() << endl;

Mat labels, centers;
cv::kmeans(data, 8, labels, cv::TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

cerr << labels.type() << " " << labels.size() << endl;
cerr << centers.type() << " " << centers.size() << endl;

cerr << centers << endl;

centers = centers.reshape(3,centers.rows);
data = data.reshape(3,data.rows);

Vec3f *p = data.ptr<Vec3f>();

for (size_t i=0; i<data.rows; i++) {
   int center_id = labels.at<int>(i);
   p[i] = centers.at<Vec3f>(center_id);
}

ocv = data.reshape(3, ocv.rows);
ocv.convertTo(ocv, CV_8U);
imshow("chili", ocv);

Mat cenviz(20,8*20,CV_8UC3);
for (int i=0; i<8; i++) {
	cenviz(Rect(i*20,0,20,20)).setTo(centers.at<Vec3f>(i));
}
imshow("centers", cenviz);
waitKey();
  	return 0;
}

