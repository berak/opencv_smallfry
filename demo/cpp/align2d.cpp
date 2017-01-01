#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

//
// Pei Lin
//

namespace PeiLin
{
    static inline Moments operator& ( const Moments & lhs, const Matx22d & rhs )
    {
        return Moments (
            lhs.m00,
            rhs ( 0, 0 ) * lhs.m10 + rhs ( 0, 1 ) * lhs.m01,
            rhs ( 1, 0 ) * lhs.m10 + rhs ( 1, 1 ) * lhs.m01,
            rhs ( 0, 0 ) * rhs ( 0, 0 ) * lhs.m20 + rhs ( 0, 1 ) * rhs ( 0, 1 ) * lhs.m02 + 2 * rhs ( 0, 0 ) * rhs ( 0, 1 ) * lhs.m11,
            rhs ( 0, 0 ) * rhs ( 1, 0 ) * lhs.m20 + rhs ( 0, 1 ) * rhs ( 1, 1 ) * lhs.m02 + ( rhs ( 0, 0 ) * rhs ( 1, 1 ) + rhs ( 0, 1 ) * rhs ( 1, 0 ) ) * lhs.m11,
            rhs ( 1, 0 ) * rhs ( 1, 0 ) * lhs.m20 + rhs ( 1, 1 ) * rhs ( 1, 1 ) * lhs.m02 + 2 * rhs ( 1, 0 ) * rhs ( 1, 1 ) * lhs.m11,
            rhs ( 0, 0 ) * rhs ( 0, 0 ) * rhs ( 0, 0 ) * lhs.m30 + 3 * rhs ( 0, 0 ) * rhs ( 0, 0 ) * rhs ( 0, 1 ) * lhs.m21 + 3 * rhs ( 0, 0 ) * rhs ( 0, 1 ) * rhs ( 0, 1 ) * lhs.m12 + rhs ( 0, 1 ) * rhs ( 0, 1 ) * rhs ( 0, 1 ) * lhs.m03,
            rhs ( 0, 0 ) * rhs ( 0, 0 ) * rhs ( 1, 0 ) * lhs.m30 + ( rhs ( 0, 0 ) * rhs ( 0, 0 ) * rhs ( 1, 1 ) + 2 * rhs ( 0, 0 ) * rhs ( 0, 1 ) * rhs ( 1, 0 ) ) * lhs.m21 + ( 2 * rhs ( 0, 0 ) * rhs ( 0, 1 ) * rhs ( 1, 1 ) + rhs ( 0, 1 ) * rhs ( 0, 1 ) * rhs ( 1, 0 ) ) * lhs.m12 + rhs ( 0, 1 ) * rhs ( 0, 1 ) * rhs ( 1, 1 ) * lhs.m03,
            rhs ( 0, 0 ) * rhs ( 1, 0 ) * rhs ( 1, 0 ) * lhs.m30 + ( rhs ( 1, 0 ) * rhs ( 1, 0 ) * rhs ( 0, 1 ) + 2 * rhs ( 0, 0 ) * rhs ( 1, 0 ) * rhs ( 1, 1 ) ) * lhs.m21 + ( 2 * rhs ( 0, 1 ) * rhs ( 1, 0 ) * rhs ( 1, 1 ) + rhs ( 1, 1 ) * rhs ( 1, 1 ) * rhs ( 0, 0 ) ) * lhs.m12 + rhs ( 0, 1 ) * rhs ( 1, 1 ) * rhs ( 1, 1 ) * lhs.m03,
            rhs ( 1, 0 ) * rhs ( 1, 0 ) * rhs ( 1, 0 ) * lhs.m30 + 3 * rhs ( 1, 0 ) * rhs ( 1, 0 ) * rhs ( 1, 1 ) * lhs.m21 + 3 * rhs ( 1, 0 ) * rhs ( 1, 1 ) * rhs ( 1, 1 ) * lhs.m12 + rhs ( 1, 1 ) * rhs ( 1, 1 ) * rhs ( 1, 1 ) * lhs.m03
        );
    }

    static inline Matx23d operator| ( const Matx22d & lhs, const Matx21d & rhs )
    {
        return Matx23d ( lhs ( 0, 0 ), lhs ( 0, 1 ), rhs ( 0 ), lhs ( 1, 0 ), lhs ( 1, 1 ), rhs ( 1 ) );
    }

    Matx23d Normalization ( InputArray I )
    {
        const Moments  M = moments ( I );
        const double  l1 = ( M.mu20 / M.m00 + M.mu02 / M.m00 + sqrt ( ( M.mu20 / M.m00 - M.mu02 / M.m00 ) * ( M.mu20 / M.m00 - M.mu02 / M.m00 ) + 4 * M.mu11 / M.m00 * M.mu11 / M.m00 ) ) / 2;
        const double  l2 = ( M.mu20 / M.m00 + M.mu02 / M.m00 - sqrt ( ( M.mu20 / M.m00 - M.mu02 / M.m00 ) * ( M.mu20 / M.m00 - M.mu02 / M.m00 ) + 4 * M.mu11 / M.m00 * M.mu11 / M.m00 ) ) / 2;
        const double  ex = ( M.mu11 / M.m00 ) / sqrt ( ( l1 - M.mu20 / M.m00 ) * ( l1 - M.mu20 / M.m00 ) + M.mu11 / M.m00 * M.mu11 / M.m00 );
        const double  ey = ( l1 - M.mu20 / M.m00 ) / sqrt ( ( l1 - M.mu20 / M.m00 ) * ( l1 - M.mu20 / M.m00 ) + M.mu11 / M.m00 * M.mu11 / M.m00 );
        const Matx22d  E = Matx22d ( ex, ey, -ey, ex );
        const double   p = min ( I.size().height, I.size().width ) / 8;
        const Matx22d  W = Matx22d ( p / sqrt ( l1 ), 0, 0, p / sqrt ( l2 ) );
        const Matx21d  c = Matx21d ( M.m10 / M.m00, M.m01 / M.m00 );
        const Matx21d  i = Matx21d ( I.size().width / 2, I.size().height / 2 );
        const Moments  N = M & W * E;
        const double  t1 = N.mu12 + N.mu30;
        const double  t2 = N.mu03 + N.mu21;
        const double phi = atan2 ( -t1, t2 );
        const double psi = ( -t1 * sin ( phi ) + t2 * cos ( phi ) >= 0 ) ? phi : ( phi + CV_PI  );
        const Matx22d  A = Matx22d ( cos ( psi ), sin ( psi ), -sin ( psi ), cos ( psi ) );
        return ( A * W * E ) | ( i - A * W * E * c );
    }
};

void alignPeiLin(const Mat &in, Mat &out)
{
    Mat bin; threshold(in,bin,60,255,1);
    Matx23d R = PeiLin::Normalization(bin);
    warpAffine(in, out, R, Size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
}

//
// PCA
//

void getOrientation(const vector<Point> &pts, Point &pos, double &angle)
{
    //Construct a buffer used by the pca analysis
    Mat data_pts = Mat(pts.size(), 2, CV_64FC1);
    for (int i = 0; i < data_pts.rows; ++i)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
    //Perform PCA analysis
    PCA pca(data_pts, Mat(), CV_PCA_DATA_AS_ROW);

    pos = Point(pca.mean.at<double>(0, 0),
                pca.mean.at<double>(0, 1));
    angle = atan2(pca.eigenvectors.at<double>(0, 1),
                  pca.eigenvectors.at<double>(0, 0));
}

vector<Point> largestContour(const Mat &in)
{
    Mat bin; threshold(in,bin,60,255,1);

    vector<vector<Point>> cts;
    findContours(bin, cts, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    double ma=0; int mi=0;
    for (size_t i = 0; i < cts.size(); ++i)
    {
        double area = contourArea(cts[i]);
        if (area < ma) continue;
        ma = area;
        mi = i;
    }
    return cts[mi];
}

void alignPCA(const Mat &in, Mat &out)
{
    vector<Point> contour = largestContour(in);

    Point pos;
    double angle;
    getOrientation(contour, pos, angle);
    cerr << "pca " << pos << " " << angle << endl;
    Mat R = getRotationMatrix2D(pos, -angle, 1.0);
    cerr << R << endl;
    Mat rot;
    warpAffine(in, out, R, Size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(127));
}

void alignDeskew(const Mat &img, Mat &out)
{
    int SZ = max(img.rows, img.cols);
    Moments m = moments(img);
    if (abs(m.mu02) < 1e-2)
    {
        out = img;
        return;
    }
    double skew = m.mu11 / m.mu02;
    cerr << "deskew " << skew << endl;
    Mat_<float> M(2,3); M << 1, skew, -0.5*SZ*skew, 0, 1, 0;
    warpAffine(img, out, M, Size(), WARP_INVERSE_MAP | INTER_LINEAR);
}

int main( int argc, char** argv )
{
    String ifile="key3.jpg";
    if (argc>1) ifile=argv[1];
    cerr << ifile << endl;
    Mat img = imread(ifile,0);
    if (img.total() > 400*400)
     resize(img,img,Size(), 0.25,0.25);

    Mat rot;
    alignPCA(img, rot);
    imshow("pca", rot);
    alignPeiLin(img, rot);
    imshow("PeiLin", rot);
    alignDeskew(img, rot);
    imshow("deskew", rot);

    /*
    RotatedRect rr = minAreaRect(cts[mi]);
    cerr << rr.angle << " " << rr.center << " " << rr.size << endl;
    Point2f pts[4];
    rr.points(pts);
    for (int i = 0; i < 4; i++) {
        line(img, pts[i], pts[(i+1)%4], Scalar(255,0));
        cerr << pts[i] << endl;
    }
    */
    imshow("img",img);
    waitKey();
    return 0;
}
