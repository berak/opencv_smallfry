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

vector<Point2d> todouble(const vector<Point> &x) {
    vector<Point2d> y;
    for (size_t i=0; i<x.size(); i++)
        y.push_back(x[i]);
    return y;
}

double distance(const vector<Point> &a, const vector<Point> &b)
{
    vector<Point2d> da = todouble(a);
    vector<Point2d> db = todouble(b);
    int xa = Mat(da).checkVector(2);
    int xb = Mat(db).checkVector(2);
    Mat F = findFundamentalMat(da, db); // requires len(da) == len(db), so i need to resample !

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

        virtual void add(const vector<Point> &p) {
            vector<Point2d> dp = todouble(p);
            Mat_<Point3d> hp;
            convertPointsHomogeneous(dp, hp);
            shapes2d.push_back(dp);
            shapes3d.push_back(hp.reshape(1,3));
        }

        virtual void match(const vector<Point> &pv, vector<Point2d> &best, double &dist, int &id){
            vector<Point2d> dp = todouble(pv);
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
    cv::Ptr<Matcher> createMatcher() {
        return cv::makePtr<MatcherImpl>();
    }
}
