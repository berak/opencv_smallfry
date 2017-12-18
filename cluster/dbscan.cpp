#include "opencv2/opencv.hpp"
#include "profile.h"
#include "cluster.h"


#include <vector>
#include <functional>

namespace dbscan
{

    static const int OUTLIER = -1;
    static const int UNDEFINED = 0;

    using DistanceMatrix = std::vector<std::vector<double>>;

    inline std::vector<int>  rangeQuery(const std::vector<Mat> &points,
                                        const double &eps,
                                        const int &q,
                                        DistanceMatrix &distanceMatrix) {
        PROFILE;
        std::vector<int> neighbors;
        for (int p = 0; p < points.size(); ++p) {
            if (q == p)
                continue;
            if (distanceMatrix[q][p] < eps) {
                neighbors.push_back(p);
            }
        }
        return neighbors;
    }

    int dbscan(const std::vector<Mat> &points, std::vector<int> &labels,
               const double &minNeighbors=3, const double &eps=1000) {
        PROFILE
        DistanceMatrix distanceMatrix(points.size(), std::vector<double>(points.size(), -1));
        { PROFILEX("dist")
            for (int i = 0; i < points.size(); ++i) {
                for (int j = i + 1; j < points.size(); ++j) {
                    distanceMatrix[i][j] = norm(points[i], points[j]);
                    distanceMatrix[j][i] = distanceMatrix[i][j];
                }
            }
        }

        labels = std::vector<int> (points.size(), UNDEFINED);
        int C = 0;
        for (int i = 0; i < points.size(); ++i) {
            PROFILEX("points")
            if (labels[i] != UNDEFINED)
                continue;
            auto neighbors = rangeQuery(points, eps, i, distanceMatrix);
            if (neighbors.size() < minNeighbors)
                continue;
            C += 1;
            labels[i] = C;
            int k = 0, j;
            while (k < neighbors.size()) {
                j = neighbors[k];
                if (labels[j] == OUTLIER)
                    labels[j] = C;
                if (labels[j] != UNDEFINED) {
                    ++k;
                    continue;
                }
                labels[j] = C;
                auto neighborJ = rangeQuery(points, eps, j, distanceMatrix);
                if (neighborJ.size() >= minNeighbors)
                    neighbors.insert(neighbors.end(), neighborJ.begin(), neighborJ.end());
                ++k;
            }
        }
        return C;
    };

	int cluster(const vector<Mat> &features, vector<int> &labels)
	{
		PROFILE;
        int n = dbscan(features, labels, 3, 1000);
		return n;
	}
}
