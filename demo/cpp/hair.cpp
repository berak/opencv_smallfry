#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

double gauss(double x, double mu, double sig2) {
    return exp(-(x-mu)*(x-mu)/(2*sig2));// / sqrt(2*CV_PI*sig2);
}

Point p(0,0);
void mouse(int e, int x, int y, int , void* )
{
	if (e) p=Point(x,y);
}
int main(int argc, char **argv)
{
	/*
	Mat a(3,3,CV_8UC3);
	Mat b(a.size(), a.depth());
	cerr << a.type() << " " << b.type() << endl;
	cvtColor(a,b,0);
	cerr << a.type() << " " << b.type() << endl;
	return 0;
	*/

	namedWindow("M");
	//setMouseCallback("M",mouse);
    string fn_haar = "haarcascade_frontalface_alt2.xml";
    CascadeClassifier haar_cascade(fn_haar);
    Mat img = imread("h4.jpg");
    resize(img,img,Size(),0.25,0.25);
    Mat gray; cvtColor(img, gray,COLOR_BGR2GRAY);
    //Mat hsv; cvtColor(img, hsv,COLOR_BGR2YCrCb);
    Mat hsv; cvtColor(img, hsv,COLOR_BGR2YUV);

    /*vector<Rect> faces;
    haar_cascade.detectMultiScale(gray, faces, 1.1, 4);
    if (faces.size() < 1) return -1;
    Rect r1 = faces[0];
    rectangle(img, r1, Scalar(200,0,0), 1);
	Rect r2(r1.x - r1.width/3, r1.y-r1.height/2, 5*r1.width/3, 2*r1.height);
    Rect r3(r1.x, r1.y-r1.height/4, r1.width, r1.height/4);
    rectangle(img, r2, Scalar(0,0,200), 1);
    rectangle(img, r3, Scalar(0,200,0), 1);
    Scalar m,d;
    meanStdDev(hsv(r3),m,d);
    cerr << m << d << endl;
    Scalar upper(m[0] + (d[0]), m[1] + (d[1]), m[2] + (d[2]));
    Scalar lower(m[0] - (d[0]), m[1] - (d[1]), m[2] - (d[2]));
	cerr << lower << upper << endl;
	Mat colMask;
	inRange(hsv, lower, upper, colMask);
	//inRange(img, Scalar(110.313, 135.931, 116.783),Scalar(200,200,200), colMask);
	*/
	Mat dd; gray.convertTo(dd,CV_64F);
	dft(dd,dd);

	int sig = 60;
	int mu_x = 120;
	int mu_y = 100;
	createTrackbar("sig","M", &sig, 400);
	createTrackbar("x","M", &mu_x, img.cols);
	createTrackbar("y","M", &mu_y, img.rows);
	while(1) {
		int sig2 = sig*sig;
		Mat_<double> kernel(dd.size());
		for (int j=0; j<img.rows; j++) {
			for (int i=0; i<img.cols; i++) {
				kernel(j,i) = gauss(j,mu_y,sig2) * gauss(i,mu_x,sig2);
			}
		}
		Mat dd2,ddr;
		multiply(dd, kernel, dd2);
		dft(dd2,ddr,DFT_INVERSE);
	    Mat ddr3;
	  	Scalar m,d;
	    meanStdDev(ddr,m,d);
	    threshold(ddr,ddr3,m[0]-d[0],255,1);
	    blur(ddr3,ddr3,Size(6,6));
	    ddr3.convertTo(ddr3, CV_8U);
		imshow("D",dd);
		imshow("R",ddr);
		imshow("R3",ddr3);
		imshow("K",kernel*6000);
		imshow("I",img);
	//	imshow("C",colMask);
	//	imshow("F",colMask & ddr3);
		if (waitKey(100) > 0) break;
	}
	return 0;
}
