#include <opencv2/core.hpp>
#include <vector>

namespace hausdorff {

int distance_min(const std::vector<cv::Point> & a, const std::vector<cv::Point>  & b)
{
    int minDistAB = 0;
    for (size_t i=0; i<a.size(); i++)
    {
        int minB = 1000000;
        for (size_t j=0; j<b.size(); j++)
        {
            int dx = (a[i].x - b[j].x);
            int dy = (a[i].y - b[j].y);
            int tmpDist = dx*dx + dy*dy;

            if (tmpDist < minB)
            {
                minB = tmpDist;
            }
            if ( tmpDist == 0 )
            {
                break; // can't get better than equal.
            }
        }
        minDistAB += minB;
    }
    return minDistAB;
}

double distance(const std::vector<cv::Point> & a, const std::vector<cv::Point> & b)
{
    int minDistAB = distance_min(a, b);
    int minDistBA = distance_min(b, a);
    int maxDist = std::max(minDistAB, minDistBA);

    return std::sqrt((double)maxDist);
}

} // namespace hausdorff


#include "matching.h"

namespace hausdorff {
    using namespace matching;

    struct MatcherImpl : Matcher {
        std::vector<std::vector<cv::Point>> shapes;

        virtual void add(const std::vector<cv::Point> &p) {
            shapes.push_back(p);
        }

        virtual void match(const std::vector<cv::Point> &pv, std::vector<cv::Point2d> &best, double &dist, int &id){
            dist=99999999;
            id=-1;
            for (size_t i=0; i<shapes.size(); i++) {
                double d = hausdorff::distance(shapes[i], pv);
                if (d < dist) {
                    dist = d;
                    id = i;
                }
            }
            if (id!=-1) { // back to point2i
                std::vector<cv::Point> &p = shapes[id];
                for (size_t i=0; i<p.size(); i++)
                    best.push_back(p[i]);
            }
        }
    };
    cv::Ptr<Matcher> createMatcher() {
        return cv::makePtr<MatcherImpl>();
    }
}
