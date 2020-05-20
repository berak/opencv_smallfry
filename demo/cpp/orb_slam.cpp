#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/calib3d.hpp>
//#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <vector>
#include <string>

using namespace cv;
using namespace std;


template<typename T>
void
projectionFromKRt(const Mat_<T> &K, const Mat_<T> &R, const Mat_<T> &t, Mat_<T> P)
{
  hconcat( K*R, K*t, P );
}

void
projectionFromKRt(InputArray K_, InputArray R_, InputArray t_, OutputArray P_)
{
  const Mat K = K_.getMat(), R = R_.getMat(), t = t_.getMat();
  const int depth = K.depth();
  CV_Assert((K.cols == 3 && K.rows == 3) && (t.cols == 1 && t.rows == 3) && (K.size() == R.size()));
  CV_Assert((depth == CV_32F || depth == CV_64F) && depth == R.depth() && depth == t.depth());

  P_.create(3, 4, depth);

  Mat P = P_.getMat();

  // type
  if( depth == CV_32F )
  {
    projectionFromKRt<float>(K, R, t, P);
  }
  else
  {
    projectionFromKRt<double>(K, R, t, P);
  }
}

template<typename T>
void
homogeneousToEuclidean(const Mat & X_, Mat & x_)
{
  int d = X_.rows - 1;

  const Mat_<T> & X_rows = X_.rowRange(0,d);
  const Mat_<T> h = X_.row(d);

  const T * h_ptr = h[0], *h_ptr_end = h_ptr + h.cols;
  const T * X_ptr = X_rows[0];
  T * x_ptr = x_.ptr<T>(0);
  for (; h_ptr != h_ptr_end; ++h_ptr, ++X_ptr, ++x_ptr)
  {
    const T * X_col_ptr = X_ptr;
    T * x_col_ptr = x_ptr, *x_col_ptr_end = x_col_ptr + d * x_.step1();
    for (; x_col_ptr != x_col_ptr_end; X_col_ptr+=X_rows.step1(), x_col_ptr+=x_.step1() )
      *x_col_ptr = (*X_col_ptr) / (*h_ptr);
  }
}

void
homogeneousToEuclidean(InputArray X_, OutputArray x_)
{
  // src
  const Mat X = X_.getMat();

  // dst
   x_.create(X.rows-1, X.cols, X.type());
  Mat x = x_.getMat();

  // type
  if( X.depth() == CV_32F )
  {
    homogeneousToEuclidean<float>(X,x);
  }
  else
  {
    homogeneousToEuclidean<double>(X,x);
  }
}

static void
triangulateDLT( const Vec2d &xl, const Vec2d &xr,
                const Matx34d &Pl, const Matx34d &Pr,
                Vec3d &point3d )
{
    Matx44d design;
    for (int i = 0; i < 4; ++i)
    {
        design(0,i) = xl(0) * Pl(2,i) - Pl(0,i);
        design(1,i) = xl(1) * Pl(2,i) - Pl(1,i);
        design(2,i) = xr(0) * Pr(2,i) - Pr(0,i);
        design(3,i) = xr(1) * Pr(2,i) - Pr(1,i);
    }

    Vec4d XHomogeneous;
    cv::SVD::solveZ(design, XHomogeneous);

    homogeneousToEuclidean(XHomogeneous, point3d);
}


/** @brief Triangulates the 3d position of 2d correspondences between n images, using the DLT
 * @param x Input vectors of 2d points (the inner vector is per image). Has to be 2xN
 * @param Ps Input vector with 3x4 projections matrices of each image.
 * @param X Output vector with computed 3d point.

 * Reference: it is the standard DLT; for derivation see appendix of Keir's thesis
 */

static void
triangulateNViews(const Mat_<double> &x, const std::vector<Matx34d> &Ps, Vec3d &X)
{
    CV_Assert(x.rows == 2);
    unsigned nviews = x.cols;
    CV_Assert(nviews == Ps.size());

    cv::Mat_<double> design = cv::Mat_<double>::zeros(3*nviews, 4 + nviews);
    for (unsigned i=0; i < nviews; ++i) {
        for(char jj=0; jj<3; ++jj)
            for(char ii=0; ii<4; ++ii)
                design(3*i+jj, ii) = -Ps[i](jj, ii);
        design(3*i + 0, 4 + i) = x(0, i);
        design(3*i + 1, 4 + i) = x(1, i);
        design(3*i + 2, 4 + i) = 1.0;
    }

    Mat X_and_alphas;
    cv::SVD::solveZ(design, X_and_alphas);
    homogeneousToEuclidean(X_and_alphas.rowRange(0, 4), X);
}


void
triangulatePoints(InputArrayOfArrays _points2d, InputArrayOfArrays _projection_matrices,
                  OutputArray _points3d)
{
    // check
    size_t nviews = (unsigned) _points2d.total();
    CV_Assert(nviews >= 2 && nviews == _projection_matrices.total());

    // inputs
    size_t n_points;
    std::vector<Mat_<double> > points2d(nviews);
    std::vector<Matx34d> projection_matrices(nviews);
    {
        std::vector<Mat> points2d_tmp;
        _points2d.getMatVector(points2d_tmp);
        n_points = points2d_tmp[0].cols;

        std::vector<Mat> projection_matrices_tmp;
        _projection_matrices.getMatVector(projection_matrices_tmp);

        // Make sure the dimensions are right
        for(size_t i=0; i<nviews; ++i) {
            CV_Assert(points2d_tmp[i].rows == 2 && points2d_tmp[i].cols == n_points);
            if (points2d_tmp[i].type() == CV_64F)
                points2d[i] = points2d_tmp[i];
            else
                points2d_tmp[i].convertTo(points2d[i], CV_64F);

            CV_Assert(projection_matrices_tmp[i].rows == 3 && projection_matrices_tmp[i].cols == 4);
            if (projection_matrices_tmp[i].type() == CV_64F)
              projection_matrices[i] = projection_matrices_tmp[i];
            else
              projection_matrices_tmp[i].convertTo(projection_matrices[i], CV_64F);
        }
    }

    // output
    _points3d.create(3, n_points, CV_64F);
    cv::Mat points3d = _points3d.getMat();

    // Two view
    if( nviews == 2 )
    {
        const Mat_<double> &xl = points2d[0], &xr = points2d[1];

        const Matx34d & Pl = projection_matrices[0];    // left matrix projection
        const Matx34d & Pr = projection_matrices[1];    // right matrix projection

        // triangulate
        for( unsigned i = 0; i < n_points; ++i )
        {
            Vec3d point3d;
            triangulateDLT( Vec2d(xl(0,i), xl(1,i)), Vec2d(xr(0,i), xr(1,i)), Pl, Pr, point3d );
            for(char j=0; j<3; ++j)
                points3d.at<double>(j, i) = point3d[j];
        }
    }
    else if( nviews > 2 )
    {
        // triangulate
        for( unsigned i=0; i < n_points; ++i )
        {
            // build x matrix (one point per view)
            Mat_<double> x( 2, nviews );
            for( unsigned k=0; k < nviews; ++k )
            {
                points2d.at(k).col(i).copyTo( x.col(k) );
            }

            Vec3d point3d;
            triangulateNViews( x, projection_matrices, point3d );
            for(char j=0; j<3; ++j)
                points3d.at<double>(j, i) = point3d[j];
        }
    }
}


struct map_points
{
    vector<Point3d> position_3d;
    vector<vector<Vec3b> > colors;
};

class OrbSLAM
{
public:
    OrbSLAM()
    {
        covisibility_graph.reserve(100);
        for(int i = 0; i < 100; i++)
            covisibility_graph[i].reserve(100);
    };
    void alignImages(Mat& im1);
    vector<KeyPoint> gridFeatureDetect(Mat& img, Mat& descriptors, int grid);
    void getMatcheBF(Mat& descriptor1, Mat& descriptor2, vector<DMatch>& matches);
    void getMatcheKNN(Mat& descriptor1, Mat& descriptor2, vector<DMatch>& matches);
    double computeScore(Mat &M, vector<Point2f>& points1, vector<Point2f>& points2, const double T_M);
    bool reconstract(Mat& H, vector<Point2f>& points1,
                     vector<Point2f>& points2);
    bool doMapInitialization(vector<Point2f>& points1, vector<Point2f>& points2);
    bool doTracking();
    void getImage(Mat &img)
    {
        frames.push_back(img);
    };
    void setCamera();
    void updateCovisibilityGraph();
protected:
    vector<Mat> gray_frames;
    vector<Mat> frames;
    vector<vector<KeyPoint> > keypoints;
    vector<Mat> descriptors;

    vector<vector<int>> covisibility_graph;
    vector<vector<Point3f> > cur_points3d;
    vector<vector<Point2f> > cur_points2d;
    vector<vector<Vec3b>>  colors;
    Mat calib_mat;
    Mat distCoeffs;
    Matx33d new_calib;
    int grid = 8;
};

void OrbSLAM::updateCovisibilityGraph()
{
    alignImages(frames.back());

    covisibility_graph.resize(covisibility_graph.size() + 1);
    for(int i = 0; i < covisibility_graph.size(); i++)
    {
        covisibility_graph[i].resize(covisibility_graph.size());
    }
    covisibility_graph[covisibility_graph.size() - 1][covisibility_graph.size() - 1] = 0;
    for(size_t i = 0; i < covisibility_graph.size() - 1; i++)
    {
        int j = covisibility_graph.size() - 1;

        cout << i << " " << j << endl;

        vector<DMatch>  matches;

        getMatcheBF(descriptors[i], descriptors[j], matches);

        covisibility_graph[i][j] = matches.size();
        //draw matches
        Mat imMatches;
        drawMatches(frames[i], keypoints[i], frames[j], keypoints[j], matches, imMatches);
        imshow("matches", imMatches);
        waitKey(10);
        vector<Point2f> points1, points2;
        colors.resize(colors.size() + 1);
        for( size_t k = 0; k < matches.size(); k++ )
        {
            points1.push_back( keypoints[i][ matches[k].queryIdx ].pt );
            points2.push_back( keypoints[j][ matches[k].trainIdx ].pt );
            colors[colors.size() - 1].push_back(frames[j].at<Vec3b>(points2[k]));
        }

        cur_points2d.resize(cur_points2d.size() + 1);
        cur_points2d[cur_points2d.size() - 1] = points2;
        cout << "there\n";
        doMapInitialization(points1, points2);
    }
}

//for logi 1080p HD C920
void OrbSLAM::setCamera()
{
    Mat calib_mat_(3, 3, CV_64FC1);
    Mat distCoeffs_ = Mat::zeros(5, 1, CV_64FC1);
    /*
    calib_mat_.at<double>(0, 0) = 987;
    calib_mat_.at<double>(0, 1) = 0.0;
    calib_mat_.at<double>(0, 2) = 630;
    calib_mat_.at<double>(1, 0) = 0.0;
    calib_mat_.at<double>(1, 1) = 987;
    calib_mat_.at<double>(1, 2) = 357;
    calib_mat_.at<double>(2, 0) = 0.0;
    calib_mat_.at<double>(2, 1) = 0.0;
    calib_mat_.at<double>(2, 2) = 1.0;
    */
    calib_mat_.at<double>(0, 0) = 612.03;
    calib_mat_.at<double>(0, 1) = 0.0;
    calib_mat_.at<double>(0, 2) = 320.15;
    calib_mat_.at<double>(1, 0) = 0.0;
    calib_mat_.at<double>(1, 1) = 661.6614;
    calib_mat_.at<double>(1, 2) = 117.5195;
    calib_mat_.at<double>(2, 0) = 0.0;
    calib_mat_.at<double>(2, 1) = 0.0;
    calib_mat_.at<double>(2, 2) = 1.0;
    calib_mat = calib_mat_;

    distCoeffs_.at<double>(0,0) = -0.128224;
    distCoeffs_.at<double>(1,0) = 0.023572;
    distCoeffs_.at<double>(2,0) = -0.0596;
    distCoeffs_.at<double>(3,0) = 0.040301;
    distCoeffs_.at<double>(4,0) = 0.0;
    distCoeffs = distCoeffs_;
    new_calib = Matx33d(612.03, 0.0, 320.15, 0.0, 661.6614,  117.5195, 0.0, 0.0, 1.0);
}

bool OrbSLAM::doMapInitialization(vector<Point2f>& points1, vector<Point2f>& points2)
{
    //vector<Point2f> points1, points2;
    //alignImages(frames.back());
    Mat h = findHomography( points1, points2, RANSAC, 5.99 );
    Mat f = findFundamentalMat(points1, points2, RANSAC, 3.84);
    double score_f = computeScore(f, points1, points2, 3.84);
    double score_h = computeScore(h, points1, points2, 5.99);
    double R_h = score_h / (score_h + score_f);
    //vector<Point3d> points3d;
    if(R_h > 0.45)
    {
        reconstract(h, points1, points2);
    }
    else
    {
        cout << "not implemented\n";
        return false;
    }
    return true;

}

void OrbSLAM::getMatcheKNN(Mat& descriptor1, Mat& descriptor2, vector<DMatch>& matches)
{
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptor1, descriptor2, knn_matches, 2 );
    const float ratio_thresh = 0.6f;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if(knn_matches[i].size() >= 2)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                matches.push_back(knn_matches[i][0]);
            }
        }
    }
}

void OrbSLAM::getMatcheBF(Mat& descriptor1, Mat& descriptor2, vector<DMatch>& matches)
{
    float good_match_percent = 0.08f;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptor1, descriptor2, matches, Mat());

    std::sort(matches.begin(), matches.end());
    const int numGoodMatches = matches.size() * good_match_percent;
    matches.erase(matches.begin()+numGoodMatches, matches.end());
}

void OrbSLAM::alignImages(Mat& im1)
{
    gray_frames.resize(gray_frames.size() + 1);
    //keypoints.resize(keypoints.size() + 1);
    descriptors.resize(descriptors.size() + 1);
    //colors.resize(colors.size() + 1)
    cvtColor(im1, gray_frames.back(), COLOR_BGR2GRAY);
    keypoints.push_back(gridFeatureDetect(gray_frames.back(), descriptors.back(), grid));

    //getMatcheBF(descriptors1, descriptors2, matches.back());

    //draw matches
    //Mat imMatches;
    //drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
    //imshow("matches", imMatches);
    //waitKey();
    //for( size_t i = 0; i < matches.back().size(); i++ )
    //{
    //    points1.push_back( keypoints1[ matches[i].queryIdx ].pt );
    //    points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
    //    colors.push_back(viz::Color(im1.at<Vec3b>(points1[i])));
    //}

}
vector<KeyPoint> OrbSLAM::gridFeatureDetect(Mat& img, Mat& descriptors, int grid)
{
    vector<KeyPoint> keypoints_;
    int step_c = img.cols / grid;

    int step_r = img.rows / grid;

    int max_threshold = 500;
    int step_threshold = 8;
    int max_features = 4000;
    max_features = max_features / grid;
    vector<Mat> masks;
    for(int i = 0; i < img.cols; i += step_c)
    {
        for(int j = 0; j < img.rows; j += step_r)
        {
            Mat mask = Mat::zeros(img.size(), CV_8UC1);
            rectangle(mask, Point(i, j), Point(i + step_c, j + step_r), Scalar(255), -1);
            masks.push_back(mask);
        }
    }

    for(size_t i = 0; i < masks.size(); i++)
    {
        Ptr<ORB> orb = ORB::create(max_features);
        orb->setScoreType(ORB::FAST_SCORE);
        for(int threshold = 20; threshold < max_threshold; threshold += step_threshold)
        {
            vector<KeyPoint> keypoints_temp;
            Mat descriptors_temp;
            orb->setFastThreshold(threshold);
            orb->detectAndCompute(img, masks[i], keypoints_temp, descriptors_temp);
            if(keypoints_temp.size() >= 5)
            {
                std::copy(begin(keypoints_temp), end(keypoints_temp), std::back_inserter(keypoints_));
                descriptors.push_back(descriptors_temp);
                break;
            }
        }
    }
cout << "grid " << keypoints_.size() << endl;
    return keypoints_;
}

//S = summ(p_M(d^2(m1, M^(-1) * m2) + p_M(d^2(m2, M * m1))))
//p_M(d^2) = 5.99 - d^2 if d^2 < 5.99
//else p_M(d^2) = 0
double OrbSLAM::computeScore(Mat &M, vector<Point2f>& points1, vector<Point2f>& points2, const double T_M)
{
    Mat M_inv = M.inv();
    Mat m2(3, points2.size(), CV_64FC1);
    for(int i = 0; i < points2.size(); i++)
    {
        m2.at<double>(0, i) = points2[i].x;
        m2.at<double>(1, i) = points2[i].y;
        m2.at<double>(2, i) = 1;
    }
    Mat M_inv_m2_mat = M_inv * m2;
    vector<Point2f> M_inv_m2;
    vector<double> dist1;
    for(int i = 0; i < points1.size(); i++)
    {
        M_inv_m2.push_back(Point2f(M_inv_m2_mat.at<double>(0, i) / M_inv_m2_mat.at<double>(2, i),
                                   M_inv_m2_mat.at<double>(1,  i) / M_inv_m2_mat.at<double>(2, i)));
        dist1.push_back((M_inv_m2[i].x - points1[i].x) * (M_inv_m2[i].x - points1[i].x) +
                        (M_inv_m2[i].y - points1[i].y) * (M_inv_m2[i].y - points1[i].y));
    }
    //TODO use convertPointsToHomogeneous() and convertPointsFromHomogeneous()
    Mat m1(3, points1.size(), CV_64FC1);
    for(int i = 0; i < points1.size(); i++)
    {
        m1.at<double>(0, i) = points1[i].x;
        m1.at<double>(1, i) = points1[i].y;
        m1.at<double>(2, i) = 0;
    }
    Mat M_m1_mat =  M * m1;
    vector<Point2f> M_m1;
    vector<double> dist2;
    double S_M = 0;
    for(int i = 0; i < points2.size(); i++)
    {
        M_m1.push_back(Point2f(M_m1_mat.at<double>(0, i) / M_m1_mat.at<double>(2, i),
                               M_m1_mat.at<double>(1, i) / M_m1_mat.at<double>(2, i)));
        dist2.push_back((M_m1[i].x - points2[i].x) * (M_m1[i].x - points2[i].x) +
                       (M_m1[i].y - points2[i].y) * (M_m1[i].y - points2[i].y));
    }
    double T_H = 5.99;
    for(int i = 0; i < dist1.size(); i++)
    {
        //cout << dist1[i] << endl;
        if(dist1[i] < T_M)
           S_M += T_H - dist1[i];
        if(dist2[i] < T_M)
           S_M += T_H - dist2[i];
    }
    cout << "S_M = " << S_M << endl;
    return S_M;
}

bool OrbSLAM::reconstract(Mat& H, vector<Point2f>& points1, vector<Point2f>& points2)
{
    vector<Mat> rotations, translations, normals;
    int solutions = decomposeHomographyMat(H, calib_mat, rotations, translations, normals);
    cout << "solution " << solutions << endl;
    if(solutions == 0)
    {
        cur_points2d.resize(cur_points2d.size() - 1);
        return false;
    }
    vector<int> possible_solution;
    filterHomographyDecompByVisibleRefpoints(rotations, normals, points1, points2, possible_solution);
    cout << "possible solution " << possible_solution.size() << endl;
    if(possible_solution.size() == 0)
    {
        cur_points2d.resize(cur_points2d.size() - 1);
        return false;
    }
    Mat projection1, projection2;
    Mat id_mat = Mat::eye(rotations[0].size(), CV_64FC1);
    Mat z = Mat::zeros(translations[0].size(), CV_64FC1);

    projectionFromKRt(calib_mat, id_mat, z, projection1);
    if(cur_points2d.size() == 1 )
        projectionFromKRt(calib_mat, rotations[possible_solution[0]], translations[possible_solution[0]], projection2);
    else
        projectionFromKRt(calib_mat, rotations[possible_solution[0]], translations[possible_solution[0]], projection2);
    cv::Mat points1Mat(2, points1.size(), CV_64FC1);
    cv::Mat points2Mat(2, points1.size(), CV_64FC1);


    for (int i=0; i < points1.size(); i++)
    {
        cv::Point2f myPoint1 = points1.at(i);
        cv::Point2f myPoint2 = points2.at(i);
        points1Mat.at<double>(0, i) = myPoint1.x;
        points1Mat.at<double>(1, i) = myPoint1.y;
        points2Mat.at<double>(0, i) = myPoint2.x;
        points2Mat.at<double>(1, i) = myPoint2.y;
    }

    vector<Mat> points2d;
    points2d.push_back(points1Mat);
    points2d.push_back(points2Mat);

    vector<Mat> projections;
    projections.push_back(projection1);
    projections.push_back(projection2);

    Mat points3d_mat(3, 1, CV_64FC1);
    Mat Rs;
    Mat Ts;
    triangulatePoints(points2d, projections, points3d_mat);


    cur_points3d.resize(cur_points3d.size() + 1);
    cout << "b" << endl;
    for(int i = 0; i < points3d_mat.cols; i++)
    {
        cur_points3d[cur_points3d.size() - 1].push_back(Point3f(points3d_mat.at<double>(0, i), points3d_mat.at<double>(1, i), points3d_mat.at<double>(2, i)));
    }
cout << "cur_points3d " << cur_points3d.size() << endl;
    return true;
}

bool OrbSLAM::doTracking()
{

    vector<Mat> rvec;        // output rotation vector
    vector<Mat> tvec;         // output translation vector
    vector<Mat> _R_matrix;   // rotation matrix
    vector<Mat> _t_matrix;   // translation matrix
    cout << cur_points2d.size() << " " << cur_points3d.size() << endl;
    for(int i = 0; i < cur_points3d.size(); i++)
    {
        rvec.push_back(Mat::zeros(3, 1, CV_64FC1));
        tvec.push_back(Mat::zeros(3, 1, CV_64FC1));
        _R_matrix.push_back(Mat::zeros(3, 3, CV_64FC1));
        _t_matrix.push_back(Mat::zeros(3, 1, CV_64FC1));
    }

    int iterationsCount = 500;        // number of Ransac iterations.
    float reprojectionError = 2.0;    // maximum allowed distance to consider it an inlier.
    float confidence = 0.95;          // RANSAC successful confidence.
    bool useExtrinsicGuess = false;
    cout << "bitch" << endl;

    int flags = 0;
    Mat inliers(Mat::zeros(1, 1, CV_64FC1));
    solvePnPRansac(cur_points3d[0], cur_points2d[0], calib_mat, distCoeffs, rvec[0], tvec[0],
                   useExtrinsicGuess, iterationsCount, reprojectionError);

/*
     iterationsCount = 500;        // number of Ransac iterations.
     reprojectionError = 2.0;    // maximum allowed distance to consider it an inlier.
     confidence = 0.95;          // RANSAC successful confidence.
     useExtrinsicGuess = false;
     solvePnPRansac(cur_points3d[1], cur_points2d[1], calib_mat, distCoeffs, rvec[1], tvec[1],
                                          useExtrinsicGuess, iterationsCount, reprojectionError);
                                          */
    //Mat  _P_matrix = cv::Mat::zeros(3, 4, CV_64FC1);   // rotation-translation matrix

    for(int i = 0; i < cur_points2d.size();i++)
    {
        Rodrigues(rvec[i],_R_matrix[i]);                   // converts Rotation Vector to Matrix
        _t_matrix[i] = tvec[i];                            // set translation matrix
    }
    vector<Affine3d> cam_pose;
    for(int i = 0; i < cur_points3d.size(); i++)
        cam_pose.push_back(Affine3d(_R_matrix[i], _t_matrix[i]));
//    viz::Viz3d window;
    vector<Point3d> all_points;
    vector<Vec3b> all_colors;
    for(int i = 0; i < cur_points3d.size(); i++)
        for(int j = 0; j < cur_points3d[i].size(); j++)
        {
            all_points.push_back(cur_points3d[i][j]);
            all_colors.push_back(colors[i][j]);
        }
/*
    //viz::WCameraPosition cpw(0.25); // Coordinate axes

    //viz::WCameraPosition cpw_frustum(new_calib, 0.3, viz::Color::yellow());
    //window.showWidget("coordinate", viz::WCoordinateSystem());
//    window.showWidget("CPW", cpw, cam_pose);

    viz::WCloud cloud_wid(all_points, all_colors);
    cloud_wid.setRenderingProperty( cv::viz::POINT_SIZE, 5 );
    window.showWidget("cameras_frames_and_lines", viz::WTrajectory(cam_pose, viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
    window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(cam_pose, new_calib, 0.1, viz::Color::yellow()));
    //window.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
    window.showWidget("points", cloud_wid);
    window.setViewerPose(cam_pose[0]);
    window.spin();
*/
    return true;
}


int main() {
    OrbSLAM slam;
    slam.setCamera();
    int count = 0;
    Mat frame;
    VideoCapture cap(0);
    while(cap.read(frame)) {
        slam.getImage(frame);
        slam.updateCovisibilityGraph();
        if (count++ > 2) {
            slam.doTracking();
        }
        imshow("I",frame);
        waitKey(10);
    }
    return 0;
}