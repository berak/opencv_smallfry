#ifndef __cluster_onboard__
#define __cluster_onboard__

#include "opencv2/opencv.hpp"
using cv::Mat;

namespace cluster {

	typedef int Cluster(const Mat &features, int K, Mat &centers);

}

namespace kmeans {
	int cluster(const Mat &features, int K, Mat &centers);
}
namespace gravi {
	int cluster(const Mat &features, int K, Mat &centers);
}
namespace flannb {
	int cluster(const Mat &features, int K, Mat &centers);
}
namespace random {
	int cluster(const Mat &features, int K, Mat &centers);
}
#endif // __cluster_onboard__
