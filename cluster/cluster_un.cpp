#include <iostream>
using namespace std;

#include "opencv2/opencv.hpp"
#include "profile.h"
#include "cluster.h"

namespace partition {
	int cluster(vector<Mat> &images, vector<int> &labels) {
		return cv::partition(images,labels,[](const Mat &a, const Mat &b){return norm(a,b)<1150;});
	}
}

int main(int argc, char **argv) {
	int SZ=32;
	vector<cv::String> fn;
	cv::glob("c:/data/faces/att/*.pgm", fn, true);
	cout << fn.size() << " files." << endl;
	vector<Mat> features;
	int id=0;
	for (auto f : fn) {
	//	cout << (id/10) << " " << f << endl;
		Mat im = cv::imread(f,0);
		if (im.empty()) continue;
		resize(im,im,cv::Size(SZ,SZ));
		im.convertTo(im, CV_32F);
		im.at<float>(0,0) = (id/10);
		features.push_back(im.reshape(1,1));
		id ++;
	}
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
	//cout << features[0] << endl;


	vector<int> labels;
	//int n = partition::cluster(features,labels);
	int n = whispers::cluster(projected,labels);
	//int n = dbscan::cluster(projected,labels);
	cout << " found " << n << " clusters and " <<  labels.size() << " labels."  << endl;
    cout << Mat(labels).t() << endl;

    for (int i=0; i<n; i++) {
    	vector<int> lb;
    	for (int j=0; j<labels.size(); j++) {
    		if (i != labels[j]) continue;
    		lb.push_back(j);
    	}
    	cout << format("%3d %3d", i, lb.size()) << " " << Mat(lb).t() << endl;
    }
    return 0;
}

