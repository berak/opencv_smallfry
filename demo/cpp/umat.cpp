#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>

using namespace cv;
using namespace std;


UMat performOpenCL(const std::string &imagePath, bool useOpenCL)
{
    cv::UMat    image(cv::USAGE_ALLOCATE_HOST_MEMORY);
    cv::UMat    output(cv::USAGE_ALLOCATE_HOST_MEMORY);

    cv::ocl::setUseOpenCL(useOpenCL);
    cv::imread(imagePath).copyTo(image);
    cv::cvtColor(image, output, cv::COLOR_BGR2HLS);
    //return output;
    return UMat();
}

int main(int argc, const char* argv[]) {
	UMat u1 = performOpenCL("img/h3.png", false);
	cerr << "1" << endl;
 	//imshow("u1", u1.getMat(0));
	UMat u2 = performOpenCL("img/h3.png", true);
	cerr << "2" << endl;
 	//imshow("u2", u2.getMat(0));
 	//waitKey();
	/*vector<Mat> imgs;
	bool ok = imreadmulti("2page.tiff",imgs,-1);
 	cerr << ok << " " << imgs.size() << endl;
 	*/

 	/*imshow("orig", image); 	waitKey(500);

	Mat redux, bars, binbar, bars2;
	reduce(image, redux, 0, REDUCE_AVG, CV_32F);
	// the raw reduce intensities, only for visualisation
	repeat(redux, image.rows, 1, bars);
 	imshow("bars", bars/255); waitKey(500);

 	// threshold and make bars of it.
	double minVal,maxVal;
	minMaxLoc(redux, &minVal, &maxVal,0,0);
	cerr << "min " << minVal << " max " << maxVal << endl;
	threshold(redux, binbar, minVal + (maxVal-minVal)*0.82, 1.0, 0);
	repeat(binbar,  image.rows, 1, bars2);
 	imshow("binbar", bars2);
 	waitKey();
 	*/
 	return 0;
}

