#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <bitset>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	theRNG().state = getTickCount();

	Mat img0 = imread("c:/data/faces/lfw40_crop/Renee_Zellweger_0006.jpg");
    Mat img1 = imread("c:/data/faces/lfw40_crop/Renee_Zellweger_0007.jpg");
    Mat img2 = imread("c:/data/faces/lfw40_crop/Richard_Gephardt_0003.jpg");

	dnn::Net net = dnn::readNetFromTorch("c:/data/mdl/openface.nn4.small2.v1.t7");
    auto proc = [&](Mat &image) {
		Mat inputBlob = dnn::blobFromImage(image, 1./255, Size(96,96), Scalar(), true, false);
		net.setInput(inputBlob);
		return net.forward().clone();
	};
	imshow("I0",img0);
	imshow("I1",img1);
	imshow("I2",img2);

	Mat f0 = proc(img0);
	Mat f1 = proc(img1);
	Mat f2 = proc(img2);
	double d0 = norm(f0,f1);
	double d1 = norm(f1,f2);
	cout << "d0 " << d0 << endl;
	cout << "d1 " << d1 << endl;
	Mat N(img2.size(), CV_32FC3,Scalar::all(0));
	Mat im2f; img2.convertTo(im2f, CV_32F);
	double db=d1;
	for (int i=0; i<10000; i++) {
		Mat n(img2.size(), CV_32FC3);
		randn(n,0,0.25);
		Mat img3 = im2f + N + n;
		img3.convertTo(img3, CV_8U);
		Mat f3 = proc(img3);
		double d = norm(f1,f3);
		cout << i << " " << d << "                  \r";
		if (d<db) {
			db=d;
			N+=n;
			double ds = sum(abs(n+N))[0]; // noise
			double dg = norm(img1,img3); // bgr distance
			cout << i << " " << d << " " << ds << " " << dg << endl;
			imshow("I",img3);
			waitKey(20);
			if (d < d0) break;
		}
	}
	cout << "best " << db << endl;
	normalize(N,N,0,1,NORM_MINMAX);
	imshow("N",N);
	waitKey();
    return 0;
}

