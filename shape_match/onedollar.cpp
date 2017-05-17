//
// mostly from: https://github.com/roxlu/ofxOneDollar/
//

#include "opencv2/opencv.hpp"
using namespace cv;

namespace onedollar {

	double length(const std::vector<Point> &points) {
		double len = 0;
		for (int i=1; i<points.size(); ++i) {
			len += norm(points[i-1] - points[i]);
		}
		return len;
	}

	Rect2d boundingBox(const std::vector<Point2d> &pts) {
		double min_x = FLT_MAX, min_y = FLT_MAX, max_x = FLT_MIN, max_y = FLT_MIN;
		std::vector<Point2d>::const_iterator it = pts.begin();
		while (it != pts.end()) {
			Point2d v = (*it);
			if(v.x < min_x) min_x = v.x;
			if(v.x > max_x) max_x = v.x;
			if(v.y < min_y) min_y = v.y;
			if(v.y > max_y) max_y = v.y;
			++it;
		}

		Rect2d rect;
		rect.x = min_x;
		rect.y = min_y;
		rect.width = (max_x - min_x);
		rect.height = (max_y - min_y);
		return rect;
	}

	void resample(const std::vector<Point> &points, int n, std::vector<Point2d> &pts) {
		double I = length(points)/(n - 1);
		double D = 0;

		for (int i = 1; i < points.size(); ++i) {
			Point2d curr = points[i];
			Point2d prev = points[i-1];
			Point2d dir = prev - curr;
			double d = norm(dir);
			if ( (D + d) >= I) {
				double qx = prev.x + ((I-D)/d) * (curr.x - prev.x);
				double qy = prev.y + ((I-D)/d) * (curr.y - prev.y);
				Point2d resampled(qx, qy);
				pts.push_back(resampled);
				D = 0.0;
			}
			else {
				D += d;
			}
		}
		// we had to do some freaky resizing because of rounding issues.
		while (pts.size() <= (n - 1)) {
			pts.push_back(points.back());
		}
		if (pts.size() > n) {
			pts.erase(pts.begin(), pts.begin()+n);
		}
	}

	Point2d centroid(const std::vector<Point2d> &pts) {
		Point2d center(0,0);
		std::vector<Point2d>::const_iterator it = pts.begin();
		while (it != pts.end()) {
			center += (*it);
			++it;
		}
		center /= double(pts.size());
		return center;
	}

	std::vector<Point2d> rotateBy(std::vector<Point2d> &pts, double nRad, const Point &c) {
		std::vector<Point2d> rotated;
		double cosa = cos(nRad);
		double sina = sin(nRad);
		std::vector<Point2d>::iterator it = pts.begin();
		while (it != pts.end()) {
			Point2d v = (*it);
			double dx = v.x - c.x;
			double dy = v.y - c.y;
			v.x = dx * cosa - dy * sina + c.x;
			v.y = dx * sina + dy * cosa + c.y;
			rotated.push_back(v);
			++it;
		}
		return rotated;
	}

	void rotateToZero(std::vector<Point2d> &pts, const Point &c) {
		double angle = (c.y - pts[0].y, c.x - pts[0].x);
		pts = rotateBy(pts, -angle, c);
	}

	void scaleTo(std::vector<Point2d> &pts, double nSize = 250.0) {
		Rect2d rect = boundingBox(pts);
		std::vector<Point2d>::iterator it = pts.begin();
		while (it != pts.end()) {
			Point2d* v = &(*it);
			v->x = v->x * (nSize/rect.width);
			v->y = v->y * (nSize/rect.height);
			++it;
		};
	}

	// translates to origin.
	void translate(std::vector<Point2d> &pts, const Point &c) {
		std::vector<Point2d>::iterator it = pts.begin();
		while (it != pts.end()) {
			Point2d* v = &(*it);
			v->x = v->x - c.x;
			v->y = v->y - c.y;
			++it;
		};
	}

	void normalize(const std::vector<Point> &points, int nNumSamples, std::vector<Point2d> &pts) {
		resample(points, nNumSamples, pts);
		Point2d c = centroid(pts);
		rotateToZero(pts, c);
		scaleTo(pts);
		translate(pts, c);
	}

	// distance between two paths.
	double pathDistance(const std::vector<Point2d> &p, const std::vector<Point2d> &q) {
		// sizes are not equal (?)
		if (p.size() != q.size()) {
			return -1.0;
		}
		double d = 0;
		for (int i = 0; i < q.size(); ++i) {
			 d += norm(p[i] - q[i]);
		}
		return d/p.size();
	}

	double distanceAtAngle(double nAngle, const std::vector<Point2d> &p, const std::vector<Point2d> &q, const Point &c) {
		std::vector<Point2d> points_tmp = p;
		points_tmp = rotateBy(points_tmp, nAngle, c);
		return pathDistance(points_tmp, q);
	}

	const double angle_precision = 1.0;
	const double golden_ratio = 0.5 * (-1.0 + sqrt(5.0));
	double distanceAtBestAngle(const std::vector<Point2d> &p, const std::vector<Point2d> &q) {
		Point2d c = centroid(p);
		double angle_range = CV_PI;
		double start_range = -angle_range;
		double end_range = angle_range;
		double x1 = golden_ratio * start_range + (1.0 - golden_ratio) * end_range;
		double f1 = distanceAtAngle(x1, p, q, c);
		double x2 = (1.0 - golden_ratio) * start_range + golden_ratio * end_range;
		double f2 = distanceAtAngle(x2, p, q, c);
		while (abs(end_range - start_range) > angle_precision) {
			if (f1 < f2) {
				end_range = x2;
				x2 = x1;
				f2 = f1;
				x1 =  golden_ratio * start_range + (1.0 - golden_ratio) * end_range;
				f1 = distanceAtAngle(x1, p, q, c);
			} else {
				start_range = x1;
				x1 = x2;
				f1 = f2;
				x2 = (1.0 - golden_ratio) * start_range + golden_ratio * end_range;
				f2 = distanceAtAngle(x2, p, q, c);
			}
		}
		return min(f1, f2);
	}

} // namespace onedollar


#include "matching.h"

namespace onedollar {
	using namespace matching;

	struct MatcherImpl : Matcher {
		int N; // resample contour size
		std::vector<std::vector<cv::Point2d>> shapes;

		MatcherImpl(int n) : N(n) {}

		virtual void add(const std::vector<cv::Point> &p) {
			std::vector<cv::Point2d> z;
			onedollar::normalize(p,N,z);
			shapes.push_back(z);
		}

		virtual void match(const std::vector<cv::Point> &p, std::vector<cv::Point2d> &best, double &dist, int &id){
			std::vector<cv::Point2d> z;
			onedollar::normalize(p,N,z);
			dist=99999999;
			id=-1;
			for (size_t i=0; i<shapes.size(); i++) {
				double d = onedollar::distanceAtBestAngle(shapes[i], z);
				if (d < dist) {
					dist = d;
					id = i;
				}
			}
			if (id!=-1) {
				best = shapes[id];
			}
		}
	};
	cv::Ptr<Matcher> createMatcher(int n) {
		return cv::makePtr<MatcherImpl>(n);
	}
}
