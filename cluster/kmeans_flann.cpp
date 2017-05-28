#include "opencv2/opencv.hpp"
#include "profile.h"
#include "cluster.h"

namespace kmeans {
	int cluster(const Mat &features, int K, Mat &dictionary) {
		PROFILE;
		Mat indices;
		cv::kmeans(features,K,indices,cv::TermCriteria(),3,cv::KMEANS_PP_CENTERS,dictionary);
		return dictionary.rows;
	}
}

// canters have to be preallocated for this.
namespace flannb {
	int cluster(const Mat &features, int K, Mat &dictionary) {
	    PROFILE;
	    cvflann::KMeansIndexParams params(K, 100, cvflann::FLANN_CENTERS_KMEANSPP);
    	return cv::flann::hierarchicalClustering<cv::flann::L2<float>>(features,dictionary,params);
    }
}
