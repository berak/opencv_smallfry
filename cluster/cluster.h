#ifndef __cluster_onboard__
#define __cluster_onboard__

#include "opencv2/opencv.hpp"
using cv::Mat;

namespace cluster {

	typedef int Cluster(const Mat &features, int K, Mat &centers);

}

// these require an initial K
namespace kmeans {
	int cluster(const Mat &features, int K, Mat &centers);
}
namespace brute {
	int cluster(const Mat &features, int K, Mat &centers);
}
namespace flannb {
	int cluster(const Mat &features, int K, Mat &centers);
}
namespace random {
	int cluster(const Mat &features, int K, Mat &centers);
}

// these don't
namespace partition {
	int cluster(const vector<Mat> &images, vector<int> &labels, double eps);
}
namespace whispers {
	int cluster(const vector<Mat> &images, vector<int> &labels, double eps);
}
namespace dbscan {
	int cluster(const vector<Mat> &images, vector<int> &labels, double eps);
}

#endif // __cluster_onboard__
