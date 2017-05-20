#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <iostream>
using std::cout;
using std::endl;
using std::vector;
using cv::Point;
using cv::Point2d;
using cv::Point3d;
using cv::Mat;
using cv::Mat_;

namespace sampson {

void resample(const vector<Point> &points, int n, vector<Point2d> &pts) {
    double D = 0, I = 0;
    for (int i=1; i<points.size(); ++i) {
        I += norm(points[i-1] - points[i]);
    }
    I /= (n - 1);

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

double distance(const vector<Point> &a, const vector<Point> &b)
{
    const int N = std::min(a.size(), b.size());
    vector<Point2d> da; resample(a,N,da);
    vector<Point2d> db; resample(b,N,db);
    int xa = Mat(da).checkVector(2);
    int xb = Mat(db).checkVector(2);
    Mat F = findFundamentalMat(da, db);

    Mat_<Point3d> ha, hb;
    convertPointsHomogeneous(da, ha);
    convertPointsHomogeneous(db, hb);
    return sampsonDistance(ha.reshape(1,3), hb.reshape(1,3), F);
}

} // namespace sampson


#include "matching.h"

namespace sampson {
    using namespace matching;

    struct MatcherImpl : Matcher {
        vector<Mat> shapes3d;
        vector<vector<Point2d>> shapes2d;
        int N;
        MatcherImpl(int n) : N(n) {}

        virtual void add(const vector<Point> &p) {
            vector<Point2d> dp;
            resample(p, N, dp);
            Mat_<Point3d> hp;
            convertPointsHomogeneous(dp, hp);
            shapes2d.push_back(dp);
            shapes3d.push_back(hp.reshape(1,3));
        }

        virtual void match(const vector<Point> &pv, vector<Point2d> &best, double &dist, int &id){
            vector<Point2d> dp;
            resample(pv, N, dp);
            Mat_<Point3d> hp;
            convertPointsHomogeneous(dp, hp);

            dist=99999999;
            id=-1;
            for (size_t i=0; i<shapes2d.size(); i++) {
                Mat F = findFundamentalMat(shapes2d[i], dp);
                double d = sampsonDistance(shapes3d[i], hp.reshape(1,3), F);
                if (d < dist) {
                    dist = d;
                    id = i;
                }
            }
            if (id != -1) { // back to point2d
                best = shapes2d[id];
            }
        }
    };

    cv::Ptr<Matcher> createMatcher(int n) {
        return cv::makePtr<MatcherImpl>(n);
    }
}
