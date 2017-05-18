#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>

namespace sampson {

std::vector<cv::Point2d> todouble(const std::vector<cv::Point> &x) {
    std::vector<cv::Point2d> y;
    for (size_t i=0; i<x.size(); i++)
        y.push_back(x[i]);
    return y;
}

double distance(const std::vector<cv::Point> &a, const std::vector<cv::Point> &b)
{
    std::vector<cv::Point2d> da = todouble(a);
    std::vector<cv::Point2d> db = todouble(b);
    cv::Mat F = findFundamentalMat(da, db);

    std::vector<cv::Point3d> ha, hb;
    cv::convertPointsHomogeneous(da, ha);
    cv::convertPointsHomogeneous(db, hb);

    return cv::sampsonDistance(ha, hb, F);
}

} // namespace sampson


#include "matching.h"

namespace sampson {
    using namespace matching;

    struct MatcherImpl : Matcher {
        std::vector<std::vector<cv::Point3d>> shapes3d;
        std::vector<std::vector<cv::Point2d>> shapes2d;

        virtual void add(const std::vector<cv::Point> &p) {
            std::vector<cv::Point2d> dp = todouble(p);
            std::vector<cv::Point3d> hp;
            cv::convertPointsHomogeneous(dp, hp);
            shapes2d.push_back(dp);
            shapes3d.push_back(hp);
        }

        virtual void match(const std::vector<cv::Point> &pv, std::vector<cv::Point2d> &best, double &dist, int &id){
            std::vector<cv::Point2d> dp = todouble(pv);
            std::vector<cv::Point3d> hp;
            cv::convertPointsHomogeneous(dp, hp);

            dist=99999999;
            id=-1;
            for (size_t i=0; i<shapes2d.size(); i++) {
                cv::Mat F = cv::findFundamentalMat(shapes2d[i], dp);
                double d = cv::sampsonDistance(shapes3d[i], hp, F);
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
