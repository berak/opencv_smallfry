#ifndef __cluster_onboard__
#define __cluster_onboard__

#include "opencv2/opencv.hpp"
using cv::Mat;

namespace cluster {

	typedef int Cluster(const Mat &features, int K, Mat &dictionary);

}

namespace kmeans {
	int cluster(const Mat &features, int K, Mat &dictionary);
}
namespace brute {
	int cluster(const Mat &features, int K, Mat &dictionary);
}
namespace flannb {
	int cluster(const Mat &features, int K, Mat &dictionary);
}
namespace random {
	int cluster(const Mat &features, int K, Mat &dictionary);
}
#endif // __cluster_onboard__
