#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


#include <vector>


namespace dbscan
{

    static const int OUTLIER = -1;
    static const int UNDEFINED = 0;

    using DistanceMatrix = std::vector<std::vector<double>>;

    double distance(const Vec4i &a, const Vec4i &b) {
        //double dx = (a[0]-b[0])*(a[0]-b[0]) + (a[2]-b[2])*(a[2]-b[2]);
        //double dy = (a[1]-b[1])*(a[1]-b[1]) + (a[3]-b[3])*(a[3]-b[3]);
        //return sqrt(min(dx,dy));
        //Vec4i c = a-b;
        return norm(a,b);
    }

    inline std::vector<int>  rangeQuery(const std::vector<Vec4i> &points,
                                        const double &eps,
                                        const int &q,
                                        DistanceMatrix &distanceMatrix) {
        std::vector<int> neighbors;
        for (int p = 0; p < points.size(); ++p) {
            if (q == p)
                continue;
            if (distanceMatrix[q][p] < eps)
                neighbors.push_back(p);
        }
        return neighbors;
    }

    int dbscan(const std::vector<Vec4i> &points, std::vector<int> &labels,
               const double &minNeighbors=3, const double &eps=1000) {

        DistanceMatrix distanceMatrix(points.size(), std::vector<double>(points.size(), -1));
        for (int i = 0; i < points.size(); ++i) {
            for (int j = i + 1; j < points.size(); ++j) {
                distanceMatrix[i][j] = distance(points[i], points[j]);
                distanceMatrix[j][i] = distanceMatrix[i][j];
            }
        }

        labels = std::vector<int> (points.size(), UNDEFINED);
        int C = 0;
        for (int i = 0; i < points.size(); ++i) {
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

    int cluster(const vector<Vec4i> &features, vector<int> &labels, double eps)
    {
        int n = dbscan(features, labels, 5, eps);
        return n;
    }
}

namespace partition {
    int cluster(vector<Vec4i> &images, vector<int> &labels, double eps) {
        return cv::partition(images,labels,[eps](const Vec4i &a, const Vec4i &b){return norm(a,b)<eps;});
    }
}

int main(int argc, char** argv)
{
    Mat im = imread("streets.png");
    Mat gr; cvtColor(im,gr,COLOR_BGR2GRAY);
   // blur(gr,gr,Size(3,3));
    erode(gr,gr,Mat());
    erode(gr,gr,Mat());
   // blur(gr,gr,Size(3,3));
    erode(gr,gr,Mat());
    //erode(gr,gr,Mat());
    erode(gr,gr,Mat());
    dilate(gr,gr,Mat());
    blur(gr,gr,Size(3,3));

    namedWindow("gr",0);
    imshow("gr",gr);
    vector<Vec4i> lines;
    HoughLinesP(gr, lines, 2, CV_PI /180, 200, 100, 5);
    cout << lines.size() << endl;
#if 0
    double dist1 = norm(lines[0],lines[1]);
    cout << "first dist " << dist1 << endl;
    cout << lines[0] << lines[1] << endl;

    vector<int> labels;
    int n = partition::cluster(lines, labels, 90);
    //int n = dbscan::cluster(lines, labels, 8);

    vector<Vec4i> centers(n);
    cout << n << " clusters found" << endl;
    for (int i=0; i<n; i++) {
        vector<int> lb;
        for (int j=0; j<labels.size(); j++) {
            if (i != labels[j]) continue;
            lb.push_back(j);
            Vec4i l = lines[labels[j]];
            /*centers[i][0] = max(centers[i][0], l[0]);
            centers[i][1] = max(centers[i][1], l[1]);
            centers[i][2] = min(centers[i][2], l[2]);
            centers[i][3] = min(centers[i][3], l[3]);*/
            centers[i] += l;
        }
        centers[i][0] /= lb.size();
        centers[i][1] /= lb.size();
        centers[i][2] /= lb.size();
        centers[i][3] /= lb.size();
        cout << format("%3d %3d", i, lb.size()) << " " << centers[i] << endl;
    }
    //cout << Mat(centers) << endl;
#endif
    Mat im2(im.size(), CV_8U, Scalar(0));
    for (int i=0; i<lines.size(); i++) { // skip "noise"
        Vec4i l=lines[i];
        line(im2, Point(l[0],l[1]), Point(l[2],l[3]), Scalar(255), 3);
    }
    blur(im2,im2,Size(5,5));
    im2 = im2>180;
    HoughLinesP(im2, lines, 2, CV_PI /180, 200, 100, 5);
    cout << lines.size() << endl;
    Mat im3(im.size(), CV_8U, Scalar(0));
    for (int i=0; i<lines.size(); i++) { // skip "noise"
        Vec4i l=lines[i];
        line(im3, Point(l[0],l[1]), Point(l[2],l[3]), Scalar(255), 1);
    }
 /*   for ( int i=0; i<3; i++) {
        blur(im2,im2,Size(5,5));
        dilate(im2,im2,Mat());
        erode(im2,im2,Mat());
        im2 = im2>40;
    }*/
    namedWindow("lines", 0);
    imshow("lines", im2);
    namedWindow("final", 0);
    imshow("final", im3);
    waitKey();
    return 0;
}
