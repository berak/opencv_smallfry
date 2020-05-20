#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

Mat showCenters(const Mat &centers , int siz=64) {
    Mat cent = centers.reshape(3, centers.rows);
cout << cent.channels() << " " << cent.type() << " " << cent.size() << " " << cent << endl;
    // make  a horizontal bar of K color patches:
    Mat draw(siz , siz * cent.rows, cent.type(), Scalar::all(0));
    for (int i=0; i<cent.rows; i++) {
         // set the resp. ROI to that value (just fill it):
         draw( Rect(i * siz, 0, siz, siz)) = cent.at<Vec3f>(i,0);
    }
    draw.convertTo(draw, CV_8U);

    // optional visualization:
    imshow("CENTERS", draw);
    waitKey();

    //imwrite("centers.png", draw);

    return draw;
}

int main()
{
	Mat img = imread("km.jpg");
	//Mat ocv; cvtColor(img,ocv,COLOR_BGR2HSV);
	//imshow("I",ocv);

    // convert to float & reshape to a [3 x W*H] Mat
    //  (so every pixel is on a row of it's own)
Mat data;
img.convertTo(data, CV_32F);
data = data.reshape(1, data.total());

// do kmeans
Mat labels, centers;
kmeans(data, 3, labels, TermCriteria(TermCriteria::COUNT, 10, 1.0), 3,
    KMEANS_PP_CENTERS, centers);
showCenters(centers);
return 0;
}
