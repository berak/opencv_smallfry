#include <iostream>
using namespace std;

#include "opencv2/opencv.hpp"
#include "profile.h"
#include "cluster.h"

namespace partition {
	int cluster(vector<Mat> &images, vector<int> &labels, double eps) {
		return cv::partition(images,labels,[eps](const Mat &a, const Mat &b){return norm(a,b)<eps;});
	}
}

int main(int argc, char **argv) {
	int SZ=32;
	vector<cv::String> fn;
	cv::glob("c:/data/faces/att/*.pgm", fn, true);
	//cv::glob("c:/data/faces/tv10/*.png", fn, true);
	cout << fn.size() << " files." << endl;
	vector<Mat> features;
	int id=0;
	for (auto f : fn) {
	//	cout << (id/10) << " " << f << endl;
		Mat im = cv::imread(f,0);
		if (im.empty()) continue;
		resize(im,im,cv::Size(SZ,SZ));
		im.convertTo(im, CV_32F);
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
	//int n = partition::cluster(features, labels, 1150);
	//int n = whispers::cluster(projected, labels, 1250);
	int n = dbscan::cluster(projected, labels, 1000); // 1000
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

