#ifndef cmt_hpp_

#define cmt_hpp_



using cv::Mat;
using cv::Point2f;
using cv::Rect;
using cv::Rect2d;
using cv::DMatch;
using cv::KeyPoint;
using cv::FeatureDetector;
using cv::DescriptorExtractor;
using cv::DescriptorMatcher;
using cv::Ptr;
using cv::RotatedRect;
using cv::Size2f;
using cv::Size2f;
using std::numeric_limits;
using std::string;
using std::vector;

namespace cmt {

float median(vector<float> & A);
Point2f rotate(const Point2f v, const float angle);
template<class T>
int sgn(T x)
{
    if (x >=0) return 1;
    else return -1;
}


class Consensus
{
public:
    Consensus() : estimate_scale(true), estimate_rotation(false), thr_cutoff(20) {};

    void initialize(const vector<Point2f> & points_normalized);
    void estimateScaleRotation(const vector<Point2f> & points, const vector<int> & classes,
            float & scale, float & rotation);
    void findConsensus(const vector<Point2f> & points, const vector<int> & classes,
            const float scale, const float rotation,
            Point2f & center, vector<Point2f> & points_inlier, vector<int> & classes_inlier);

    bool estimate_scale;
    bool estimate_rotation;

private:
    float thr_cutoff;
    vector<Point2f> points_normalized;
    Mat distances_pairwise;
    Mat angles_pairwise;
};

class Fusion
{
public:
    void preferFirst(const vector<Point2f> & firstPoints, const vector<int> & firstClasses,
           const vector<Point2f> & secondPoints, const vector<int> & secondClasses,
           vector<Point2f> & fusedPoints, vector<int> & fusedClasses);
};

class Matcher
{
public:
    Matcher() : thr_dist(0.25), thr_ratio(0.8), thr_cutoff(20) {};
    void initialize(const vector<Point2f> & pts_fg_norm, const Mat desc_fg, const vector<int> & classes_fg,
            const Mat desc_bg, const Point2f center);
    void matchGlobal(const vector<KeyPoint> & keypoints, const Mat descriptors,
            vector<Point2f> & points_matched, vector<int> & classes_matched);
    void matchLocal(const vector<KeyPoint> & keypoints, const Mat descriptors,
            const Point2f center, const float scale, const float rotation,
            vector<Point2f> & points_matched, vector<int> & classes_matched);

private:
    vector<Point2f> pts_fg_norm;
    Mat database;
    vector<int> classes;
    int desc_length;
    int num_bg_points;
    Ptr<DescriptorMatcher> bfmatcher;
    float thr_dist;
    float thr_ratio;
    float thr_cutoff;
};

class Tracker
{
public:
    Tracker() : thr_fb(30) {};
    void track(const Mat im_prev, const Mat im_gray, const vector<Point2f> & points_prev,
            vector<Point2f> & points_tracked, vector<unsigned char> & status);

private:
    float thr_fb;
};




class CMT
{
public:
    CMT() : str_detector("FAST"), str_descriptor("BRISK") {};
    void initialize(const Mat im_gray, const Rect rect);
    void processFrame(const Mat im_gray);

    Fusion fusion;
    Matcher matcher;
    Tracker tracker;
    Consensus consensus;

    string str_detector;
    string str_descriptor;

    vector<Point2f> points_active; //public for visualization purposes
    RotatedRect bb_rot;

private:
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> descriptor;

    Size2f size_initial;

    vector<int> classes_active;

    float theta;

    Mat im_prev;
};



}

#endif // cmt_hpp_
