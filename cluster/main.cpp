#include <iostream>
using namespace std;

#include "opencv2/opencv.hpp"
#include "profile.h"
#include "cluster.h"

namespace kmeans {
	int cluster(const Mat &features, int K, Mat &centers) {
		PROFILE;
		Mat indices;
		cv::kmeans(features,K,indices,cv::TermCriteria(),3,cv::KMEANS_PP_CENTERS,centers);
		return centers.rows;
	}
}

namespace flannb {
	int cluster(const Mat &features, int K, Mat &centers) {
	    PROFILE;
	    cvflann::KMeansIndexParams params(K, 100, cvflann::FLANN_CENTERS_KMEANSPP);
    	return cv::flann::hierarchicalClustering<cv::flann::L2<float>>(features,centers,params);
    }
}

struct Cluster {
	vector<int> ids;
	Mat cen;
	int id;

	Mat rowsum(const Mat &features) const {
		PROFILE
		Mat s = Mat(1, features.cols, CV_32F, 0.0f);
		for (size_t i=0; i<ids.size(); i++) {
			s += features.row(ids[i]);
		}
		return s;
	}
	Mat recenter(const Mat &features) {
		PROFILE
		cen = rowsum(features);
		cen /= ids.size();
		return cen;
	}
	float meanEnergy(const Mat &features) const {
		float e=0;
		for (size_t i=0; i<ids.size(); i++) {
			Mat f = features.row(ids[i]);
			e += norm(f,cen);
		}
		e /= ids.size();
		return e;
	}
};

double meanEnergy(const Mat &features, const std::vector<Cluster> &clusters) {
	float e=0;
	for (int k=0; k<clusters.size(); k++) {
		e += clusters[k].meanEnergy(features);
	}
	e /= clusters.size();
	return e;
}

namespace random {
	void init(const Mat &features, vector<Cluster> &cl) {
		PROFILE;
		cv::RNG rng(5237854);
		for (size_t i=0; i<features.rows; i++) {
			size_t id = rng.uniform(0, cl.size());
			Cluster & c = cl[id];
			c.id = id;
			c.ids.push_back(i);
		}
		for (vector<Cluster>::iterator it = cl.begin(); it != cl.end(); it++ ) {
			it->recenter(features);
		}
	}
	int cluster(const Mat &features, int K, Mat &centers) {
		PROFILE;
		cv::RNG rng(5237854);
		vector<Cluster> cl(K);
		init(features, cl);
		//double e1=meanEnergy(features, cl);
		//cout << "mean energy " << e1 << endl;

		const int ATTEMPTS = 1e6;
		int moved = 0, nomove = 0, i=0;
		for (; i<ATTEMPTS; i++) {
			size_t l = rng.uniform(0, cl.size());
			size_t r = rng.uniform(0, cl.size());
			CV_Assert(cl[l].ids.size()>0);
			size_t m = rng.uniform(0, cl[l].ids.size());
			size_t lf = cl[l].ids[m];
			double a = norm(features.row(lf), cl[l].cen);
			double b = norm(features.row(lf), cl[r].cen);
			if (a > b) {
				cl[r].ids.push_back(lf);
				cl[l].ids.erase(cl[l].ids.begin() + m);
				cl[r].recenter(features);
				cl[l].recenter(features);
				moved ++;
				nomove=0;
				//cout << i << " " << l << " " << r << " " << m << " " << lf << "	 " << a << " " << b <<endl;
			} else {
				if (++nomove > 10000)
					break;;
			}
		}
		cout << i << " gens, " << moved << " items moved." << endl;
		//double e2 = meanEnergy(features, cl);
		//cout << "mean energy " << e2 << endl;

		centers.release();
		for (int k=0; k<K; k++) {
			centers.push_back(cl[k].cen.reshape(1,1));
		}
		return centers.rows;
	}
}

namespace brute {
	int cluster(const Mat &features, int K, Mat &centers) {
		PROFILE
		vector<Cluster> cl(K);
		random::init(features,cl);
		//double e1=meanEnergy(features, cl);
		//cout << "mean energy " << e1 << endl;

		const int ATTEMPTS = 1e5;
		int moved = 0, nomove = 0, i=0, run=1;
		for (; i<ATTEMPTS && run; i++) {
			for (size_t r=0; (r<cl.size()) && run; r++) {
				for (size_t l=0; (l<cl.size()) && run; l++) {
					if (l==r) continue;
					for (size_t m=0; m<cl[l].ids.size() && run; m++) {
						size_t lf = cl[l].ids[m];
						double a = norm(features.row(lf), cl[l].cen,NORM_L1);
						double b = norm(features.row(lf), cl[r].cen,NORM_L1	);
						if (a > b) {
							cl[r].ids.push_back(lf);
							cl[l].ids.erase(cl[l].ids.begin() + m);
							moved ++;
							nomove = 0;
							//cout << i << " " << l << " " << r << " " << m << " " << lf << "	 " << a << " " << b <<endl;
							cl[r].recenter(features);
							cl[l].recenter(features);
						} else {
							nomove ++;
							if (nomove >= ATTEMPTS/10)
								run = false;
						}
					}
				}
			}
		}
		cout << i << " gens, " << moved << " items moved." << endl;
		//double e2 = meanEnergy(features, cl);
		//cout << "mean energy " << e2 << endl;

		centers.release();
		for (int k=0; k<K; k++) {
			centers.push_back(cl[k].cen.reshape(1,1));
		}
		return centers.rows;
	}
}

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
	if (n<K) cout << n << " centers of " << K << endl;
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

