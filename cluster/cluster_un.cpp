#include <iostream>
using namespace std;

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "profile.h"
#include "cluster.h"

namespace partition {
	int cluster(vector<Mat> &images, vector<int> &labels, double eps) {
		return cv::partition(images,labels,[eps](const Mat &a, const Mat &b){return norm(a,b)<eps;});
	}
}

int main(int argc, char **argv) {
	int SZ=96;
	vector<cv::String> fn;
	//cv::glob("c:/data/faces/att/*.pgm", fn, true);
	cv::glob("c:/data/faces/lfw40_crop/*.jpg", fn, false);
	//cv::glob("c:/data/faces/tv10/*.png", fn, true);
	cout << fn.size() << " files." << endl;
	vector<Mat> features;
	for (auto f : fn) {
		Mat im = cv::imread(f,1);
		if (im.empty()) continue;
		resize(im,im,cv::Size(SZ,SZ));
		//im.convertTo(im, CV_32F);
		features.push_back(im);
	}
	/*
	Mat pcafeat;
	for (auto m: features)
		pcafeat.push_back(m);
	PCA pca(pcafeat,Mat(),0,0);
	vector<Mat> projected;
	for (int r=0; r<pcafeat.rows; r++)
		projected.push_back(pca.project(features[r]));
	cout << features[0].size() << endl;
	cout << norm(features[0],features[1]) << endl;
	cout << norm(features[0],features[30]) << endl;
	*/
	vector<Mat> projected;
    dnn::Net net = dnn::readNetFromTorch("c:/data/openface.nn4.small2.v1.t7");
    for (auto f: features) {
	    Mat inputBlob = dnn::blobFromImage(f, 1./255, Size(SZ, SZ), Scalar(), true, false);   //Convert Mat to image batch
	    net.setInput(inputBlob);
		Mat vec = net.forward();
		//cout << vec.size[0] << " "<< vec.size[1] << " " << vec.size[2] << " "<< vec.size[3] << endl;
		projected.push_back(vec);
//		cout << vec << endl;
    }
    double md=0;
    double MD=0;
    int k=0;
    for (int i=0; i<400; i++) {
    	for (int j=i+1; j<400; j++,k++) {
    		double d = norm(projected[i], projected[j]);
    		cout << i << " " << j << " " << d << endl;
    		md += d;
    		MD = std::max(d,MD);
    	}
    }
    md /= k;
	cout << md << " " << MD << endl;

	vector<int> labels;
	//int n = partition::cluster(features, labels, 1150);
	int n = whispers::cluster(projected, labels, 0.6);//1250);
	//int n = dbscan::cluster(projected, labels, 0.6); // 1000
	cout << " found " << n << " clusters and " <<  labels.size() << " labels."  << endl;

    float err = 0;
    for (int i=0; i<n; i++) {
    	vector<int> lb;
 		Mat stats(1, 40, CV_32S, Scalar(0));
    	for (int j=0; j<labels.size(); j++) {
    		if (i != labels[j]) continue;
    		lb.push_back(j);
    		int id = j/10;
    		stats.at<int>(id) ++;
    	}
    	Point p; double d;
    	minMaxLoc(stats, 0, &d, 0, &p);
    	int missed = abs(std::max(10, int(lb.size())) - int(d));
    	err += missed;
    	cout << format("%3d %3d %3d %3d %3d", i, lb.size(), int(d), p.x, missed) << " " << Mat(lb).t() << endl;
    }
    cout << "error: " << err / 400 << endl;;
    return 0;
}

