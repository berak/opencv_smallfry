#ifndef __matching_onboard__
#define __matching_onboard__
#include <opencv2/core.hpp>


namespace matching {

	//! One Against Many
	//! if we constantly compare incoming objects against a fixed db,
	//!    we can save a lot by caching the features in target space.
	struct Matcher {
		//! cache "Gallery"
		virtual void add(const std::vector<cv::Point> &pv) = 0;
		//! find closest, return (possibly normalized) shape, dist and id.
		virtual void match(const std::vector<cv::Point> &pv, std::vector<cv::Point2d> &best, double &dist, int &id) = 0;
	};

	//! dissimilarity
	typedef double Distance(const std::vector<cv::Point> &, const std::vector<cv::Point> &);
}


//
// implementations
//   all of them have a "distance" method for 2 contours/shapes/gestures
//   and an optimized "match against a gallery" object
//

namespace fourier {
	double distance(const std::vector<cv::Point> &a, const std::vector<cv::Point> &b);
	cv::Ptr<matching::Matcher> createMatcher(int nResample=128);
}

namespace onedollar {
	double distance(const std::vector<cv::Point> &a, const std::vector<cv::Point> &b);
	cv::Ptr<matching::Matcher> createMatcher(int nResample=128);
}

namespace hausdorff {
	double distance(const std::vector<cv::Point> &a, const std::vector<cv::Point> &b);
	cv::Ptr<matching::Matcher> createMatcher();
}

namespace sampson {
	double distance(const std::vector<cv::Point> &a, const std::vector<cv::Point> &b);
	cv::Ptr<matching::Matcher> createMatcher();
}

#endif // __matching_onboard__
