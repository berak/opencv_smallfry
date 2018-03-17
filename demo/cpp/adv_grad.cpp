#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <bitset>

using namespace cv;
using namespace std;
/*struct tan_h {
	Mat fw(const Mat &m, float f=1)
	{
	    PROFILE;
	    Mat _m,ep,en,sp,sn,res;
	    multiply(m, -f, _m);
	    exp(m, ep);
	    exp(_m, en);
	    add(ep, en, sp);
	    subtract(ep, en, sn);
	    divide(sn, sp, res);
	    return res;
	}
	Mat bw(const Mat &m)
	{
	    PROFILE;
	    Mat _m, res;
	    subtract(1, m, _m);
	    sqrt(_m, res);
	    return res;
	}
};
*/
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

	Mat W(128, img2.total()*3, CV_32F);
	randn(W,0,0.1);
	Mat n(img2.size(), CV_32FC3);
	randn(n,0,0.1);
	Mat img3,im2f; img2.convertTo(im2f, CV_32F);

	for (int i=0; i<10000; i++) {
		img3 = im2f + n;
		img3.convertTo(img3, CV_8U);
		Mat f3 = proc(img3);
		Mat loss = f1 - f3;
		Mat u = n.reshape(1,1);
		//cout << "loss " << loss.size() << " " << loss.type() << endl;
		//cout << "u    " << u.size() << " " << u.type() << endl;
		//cout << "W    " << W.size() << " " << W.type() << endl;
		Mat g = loss.t() * u;
		//cout << "g    " << g.size() << " " << g.type() << endl;

		W -= 0.01 * g;
		n = f1 * W;
		//cout << "n    " << n.size() << " " << n.type() << endl;

		n = n.reshape(3,96);
		//cout << n.size() << n.type() << endl;

		double dg = norm(f1,f3); // bgr distance
		cout << i << " " << dg << " " << sum(g)[0] << " " << sum(W)[0] << endl;
		imshow("I",img3);
		normalize(n,n,0,1,NORM_MINMAX);
		imshow("N",n);
		waitKey(5);
	}
	//cout << "best " << db << endl;
	//normalize(N,N,0,1,NORM_MINMAX);
	//imshow("N",n);
	waitKey();
    return 0;
}

