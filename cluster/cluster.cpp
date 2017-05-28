#include <iostream>
using namespace std;

#include "opencv2/opencv.hpp"
#include "profile.h"
#include "cluster.h"


int main(int argc, char **argv) {
	cv::String meth = argc>1 ? argv[1] : "kmeans";
	int K2 = argc>2 ? atoi(argv[2]) : 4;
	int SZ = argc>3 ? atoi(argv[3]) : 32;
	int K = K2*K2;

	cluster::Cluster *Cluster;
	if (meth=="kmeans") Cluster = &kmeans::cluster;
	if (meth=="flann")  Cluster = &flannb::cluster;
	if (meth=="random") Cluster = &random::cluster;
	if (meth=="brute")  Cluster = &brute::cluster;

	vector<cv::String> fn;
	cv::glob("c:/data/faces/att/*.pgm", fn, true);
	cout << fn.size() << " files." << endl;
	Mat features;
	for (auto f : fn) {
		Mat im = cv::imread(f,0);
		if (im.empty()) continue;
		resize(im,im,cv::Size(SZ,SZ));
		im.convertTo(im, CV_32F);
		features.push_back(im.reshape(1,1));
	}

	cout << "start cluster " << meth << " " << K << endl;
	Mat cen = Mat::zeros(K, features.cols, CV_32F); // needs to be persistant for flann
	int n = Cluster(features,K,cen);
	if (n<K) cout << n << " dictionary of " << K << endl;

	Mat draw(SZ*K2,SZ*K2,CV_8U);
	for (int i=0,k=0; i<K2; i++) {
		for (int j=0; (j<K2) && (k<n); j++,k++) {
			Mat trg = draw(cv::Rect(i*SZ,j*SZ,SZ,SZ));
			Mat src = cen.row(k).reshape(1,SZ);
			src.convertTo(src,CV_8U);
			src.copyTo(trg);
		}
	}
	cv::namedWindow(meth,0);
    cv::imshow(meth,draw);
    cv::waitKey();
    return 0;
}

