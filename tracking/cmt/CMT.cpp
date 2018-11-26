
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/video.hpp>
#include "fastcluster/fastcluster.h"
#include "cmt.hpp"

using std::max_element;
using std::distance;

class TrackerCMT : public cv::Tracker {

    cmt::CMT cmt;
    struct DummyModel : cv::TrackerModel
    {
         virtual void modelUpdateImpl() CV_OVERRIDE {}
         virtual void modelEstimationImpl( const std::vector<Mat>& ) CV_OVERRIDE {}
    };

    Mat cvt(const Mat image) {
        Mat gray;
        if (image.channels()>1) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }
        return gray;
    }

public:

    virtual bool initImpl(const Mat& image, const Rect2d& boundingBox)
    {
        model = cv::makePtr<DummyModel>();
        cmt.initialize(cvt(image),boundingBox);
        return true;
    }
    virtual bool updateImpl(const Mat& image, Rect2d& boundingBox)
    {
        cmt.processFrame(cvt(image));
        boundingBox = cmt.bb_rot.boundingRect();
        return true;
    }
    virtual void write( cv::FileStorage& fs ) const {}
    virtual void read( const cv::FileNode& fn ){}

};

cv::Ptr<cv::Tracker> createTrackerCMT() { return cv::makePtr<TrackerCMT>(); }

namespace cmt {


void CMT::initialize(const Mat im_gray, const Rect rect)
{
    //Remember initial size
    size_initial = rect.size();

    //Remember initial image
    im_prev = im_gray;

    //Compute center of rect
    Point2f center = Point2f(rect.x + rect.width/2.0, rect.y + rect.height/2.0);

    //Initialize rotated bounding box
    bb_rot = RotatedRect(center, size_initial, 0.0);

    //Initialize detector and descriptor
#if CV_MAJOR_VERSION > 2
    detector = cv::FastFeatureDetector::create();
    descriptor = cv::BRISK::create();
#else
    detector = FeatureDetector::create(str_detector);
    descriptor = DescriptorExtractor::create(str_descriptor);
#endif

    //Get initial keypoints in whole image and compute their descriptors
    vector<KeyPoint> keypoints;
    detector->detect(im_gray, keypoints);

    //Divide keypoints into foreground and background keypoints according to selection
    vector<KeyPoint> keypoints_fg;
    vector<KeyPoint> keypoints_bg;

    for (size_t i = 0; i < keypoints.size(); i++)
    {
        KeyPoint k = keypoints[i];
        Point2f pt = k.pt;

        if (pt.x > rect.x && pt.y > rect.y && pt.x < rect.br().x && pt.y < rect.br().y)
        {
            keypoints_fg.push_back(k);
        }

        else
        {
            keypoints_bg.push_back(k);
        }

    }

    //Create foreground classes
    vector<int> classes_fg;
    classes_fg.reserve(keypoints_fg.size());
    for (size_t i = 0; i < keypoints_fg.size(); i++)
    {
        classes_fg.push_back(i);
    }

    //Compute foreground/background features
    Mat descs_fg;
    Mat descs_bg;
    descriptor->compute(im_gray, keypoints_fg, descs_fg);
    descriptor->compute(im_gray, keypoints_bg, descs_bg);

    //Only now is the right time to convert keypoints to points, as compute() might remove some keypoints
    vector<Point2f> points_fg;
    vector<Point2f> points_bg;

    for (size_t i = 0; i < keypoints_fg.size(); i++)
    {
        points_fg.push_back(keypoints_fg[i].pt);
    }


    for (size_t i = 0; i < keypoints_bg.size(); i++)
    {
        points_bg.push_back(keypoints_bg[i].pt);
    }

    //Create normalized points
    vector<Point2f> points_normalized;
    for (size_t i = 0; i < points_fg.size(); i++)
    {
        points_normalized.push_back(points_fg[i] - center);
    }

    //Initialize matcher
    matcher.initialize(points_normalized, descs_fg, classes_fg, descs_bg, center);

    //Initialize consensus
    consensus.initialize(points_normalized);

    //Create initial set of active keypoints
    for (size_t i = 0; i < keypoints_fg.size(); i++)
    {
        points_active.push_back(keypoints_fg[i].pt);
        classes_active.push_back(classes_fg[i]); // #42
    }
}

void CMT::processFrame(Mat im_gray) {


    //Track keypoints
    vector<Point2f> points_tracked;
    vector<unsigned char> status;
    tracker.track(im_prev, im_gray, points_active, points_tracked, status);


    //keep only successful classes
    vector<int> classes_tracked;
    for (size_t i = 0; i < classes_active.size(); i++)
    {
        if (status[i])
        {
            classes_tracked.push_back(classes_active[i]);
        }

    }

    //Detect keypoints, compute descriptors
    vector<KeyPoint> keypoints;
    detector->detect(im_gray, keypoints);


    Mat descriptors;
    descriptor->compute(im_gray, keypoints, descriptors);

    //Match keypoints globally
    vector<Point2f> points_matched_global;
    vector<int> classes_matched_global;
    matcher.matchGlobal(keypoints, descriptors, points_matched_global, classes_matched_global);


    //Fuse tracked and globally matched points
    vector<Point2f> points_fused;
    vector<int> classes_fused;
    fusion.preferFirst(points_tracked, classes_tracked, points_matched_global, classes_matched_global,
            points_fused, classes_fused);


    //Estimate scale and rotation from the fused points
    float scale;
    float rotation;
    consensus.estimateScaleRotation(points_fused, classes_fused, scale, rotation);


    //Find inliers and the center of their votes
    Point2f center;
    vector<Point2f> points_inlier;
    vector<int> classes_inlier;
    consensus.findConsensus(points_fused, classes_fused, scale, rotation,
            center, points_inlier, classes_inlier);


    //Match keypoints locally
    vector<Point2f> points_matched_local;
    vector<int> classes_matched_local;
    matcher.matchLocal(keypoints, descriptors, center, scale, rotation, points_matched_local, classes_matched_local);

    double ratio = double(points_active.size()) / keypoints.size();
    printf("%d %d %f\n", points_active.size() , keypoints.size(), ratio);
    //Clear active points
    points_active.clear();
    classes_active.clear();

    //Fuse locally matched points and inliers
    fusion.preferFirst(points_matched_local, classes_matched_local, points_inlier, classes_inlier, points_active, classes_active);
//    points_active = points_fused;
//    classes_active = classes_fused;


    //TODO: Use theta to suppress result
    bb_rot = RotatedRect(center,  size_initial * scale, rotation/CV_PI * 180);

    //Remember current image
    im_prev = im_gray;
}


//TODO: Check for even/uneven number of elements
//The order of the elements of A is changed
float median(vector<float> & A)
{

    if (A.size() == 0)
    {
        return numeric_limits<float>::quiet_NaN();
    }

    nth_element(A.begin(), A.begin() + A.size()/2, A.end());

    return A[A.size()/2];
}

Point2f rotate(const Point2f v, const float angle)
{
    Point2f r;
    r.x = cos(angle) * v.x - sin(angle) * v.y;
    r.y = sin(angle) * v.x + cos(angle) * v.y;

    return r;
}


void Consensus::initialize(const vector<Point2f> & points_normalized)
{

    //Copy normalized points
    this->points_normalized = points_normalized;

    size_t num_points = points_normalized.size();

    //Create matrices of pairwise distances/angles
    distances_pairwise = Mat(num_points, num_points, CV_32FC1);
    angles_pairwise = Mat(num_points, num_points, CV_32FC1);

    for (size_t i = 0; i < num_points; i++)
    {
        for (size_t j = 0; j < num_points; j++)
        {
            Point2f v = points_normalized[i] - points_normalized[j];

            float distance = norm(v);
            float angle = atan2(v.y,v.x);

            distances_pairwise.at<float>(i,j) = distance;
            angles_pairwise.at<float>(i,j) = angle;
        }

    }
}


//TODO: Check for estimate_scale, estimate_rotation
void Consensus::estimateScaleRotation(const vector<Point2f> & points, const vector<int> & classes,
        float & scale, float & rotation)
{

    //Compute pairwise changes in scale/rotation
    vector<float> changes_scale;
    if (estimate_scale) changes_scale.reserve(points.size()*points.size());
    vector<float> changes_angles;
    if (estimate_rotation) changes_angles.reserve(points.size()*points.size());

    for (size_t i = 0; i < std::min(points.size(),classes.size()); i++) // #46
    {
        for (size_t j = 0; j < std::min(points.size(),classes.size()); j++) // #46
        {
            if (classes[i] != classes[j])
            {
                Point2f v = points[i] - points[j];

                if (estimate_scale)
                {
                    float distance = norm(v);
                    float distance_original = distances_pairwise.at<float>(classes[i],classes[j]);
                    float change_scale = distance / distance_original;
                    changes_scale.push_back(change_scale);
                }

                if (estimate_rotation)
                {
                    float angle = atan2(v.y,v.x);
                    float angle_original = angles_pairwise.at<float>(classes[i],classes[j]);
                    float change_angle = angle - angle_original;

                    //Fix long way angles
                    if (fabs(change_angle) > CV_PI) {
                        change_angle = sgn(change_angle) * 2 * CV_PI + change_angle;
                    }

                    changes_angles.push_back(change_angle);
                }
            }

        }

    }

    //Do not use changes_scale, changes_angle after this point as their order is changed by median()
    if (changes_scale.size() < 2) scale = 1;
    else scale = median(changes_scale);

    if (changes_angles.size() < 2) rotation = 0;
    else rotation = median(changes_angles);

}

void Consensus::findConsensus(const vector<Point2f> & points, const vector<int> & classes,
        const float scale, const float rotation,
        Point2f & center, vector<Point2f> & points_inlier, vector<int> & classes_inlier)
{

    //If no points are available, return nan
    if (points.size() < 2) // #31 (needs at least 2 points)
    {
        center.x = numeric_limits<float>::quiet_NaN();
        center.y = numeric_limits<float>::quiet_NaN();


        return;
    }

    //Compute votes
    vector<Point2f> votes(points.size());
    for (size_t i = 0; i < std::min(points.size(),classes.size()); i++) // #46
    {
        votes[i] = points[i] - scale * rotate(points_normalized[classes[i]], rotation);
    }

    size_t N = points.size();

    float * D = new float[N*(N-1)/2]; //This is a lot of memory, so we put it on the heap
    cluster_result Z(N-1);

    //Compute pairwise distances between votes
    int index = 0;
    for (size_t i = 0; i < std::min(points.size(),votes.size()); i++) // #46
    {
        for (size_t j = i+1; j < std::min(points.size(),votes.size()); j++) // #46
        {
            //TODO: This index calculation is correct, but is it a good thing?
            //int index = i * (points.size() - 1) - (i*i + i) / 2 + j - 1;
            D[index] = norm(votes[i] - votes[j]);
            index++;
        }
    }

    MST_linkage_core(N,D,Z);

    union_find nodes(N);

    //Sort linkage by distance ascending
    std::stable_sort(Z[0], Z[N-1]);

    //S are cluster sizes
    int * S = new int[2*N-1];
    //TODO: Why does this loop go to 2*N-1? Shouldn't it be simply N? Everything > N gets overwritten later
    for(int i = 0; i < 2*N-1; i++)
    {
        S[i] = 1;
    }

    t_index parent = 0; //After the loop ends, parent contains the index of the last cluster
    for (node const * NN=Z[0]; NN!=Z[N-1]; ++NN)
    {
        // Get two data points whose clusters are merged in step i.
        // Find the cluster identifiers for these points.
        t_index node1 = nodes.Find(NN->node1);
        t_index node2 = nodes.Find(NN->node2);

        // Merge the nodes in the union-find data structure by making them
        // children of a new node
        // if the distance is appropriate
        if (NN->dist < thr_cutoff)
        {
            parent = nodes.Union(node1, node2);
            S[parent] = S[node1] + S[node2];
        }
    }

    //Get cluster labels
    int * T = new int[N];
    for (t_index i = 0; i < N; i++)
    {
        T[i] = nodes.Find(i);
    }

    //Find largest cluster
    int S_max = distance(S, max_element(S, S + 2*N-1));

    //Find inliers, compute center of votes
    points_inlier.reserve(S[S_max]);
    classes_inlier.reserve(S[S_max]);
    center.x = center.y = 0;

    for (size_t i = 0; i < std::min(points.size(),classes.size()); i++) // #46
    {
        //If point is in consensus cluster
        if (T[i] == S_max)
        {
            points_inlier.push_back(points[i]);
            classes_inlier.push_back(classes[i]);
            center.x += votes[i].x;
            center.y += votes[i].y;
        }

    }

    center.x /= points_inlier.size();
    center.y /= points_inlier.size();

    delete[] D;
    delete[] S;
    delete[] T;
}


void Fusion::preferFirst(const vector<Point2f> & points_first, const vector<int> & classes_first,
    const vector<Point2f> & points_second, const vector<int> & classes_second,
    vector<Point2f> & points_fused, vector<int> & classes_fused)
{

    points_fused = points_first;
    classes_fused = classes_first;

    for (size_t i = 0; i < points_second.size(); i++)
    {
        int class_second = classes_second[i];

        bool found = false;
        for (size_t j = 0; j < points_first.size(); j++)
        {
            int class_first = classes_first[j];
            if (class_first == class_second) found = true;
        }

        if (!found)
        {
            points_fused.push_back(points_second[i]);
            classes_fused.push_back(class_second);
        }

    }
}




void Matcher::initialize(const vector<Point2f> & pts_fg_norm, const Mat desc_fg, const vector<int> & classes_fg,
        const Mat desc_bg, const Point2f center)
{

    //Copy normalized points
    this->pts_fg_norm = pts_fg_norm;

    //Remember number of background points
    this->num_bg_points = desc_bg.rows;

    //Form database by stacking background and foreground features
    if (desc_bg.rows > 0 && desc_fg.rows > 0)
        vconcat(desc_bg, desc_fg, database);
    else if (desc_bg.rows > 0)
        database = desc_bg;
    else
        database = desc_fg;

    //Extract descriptor length from features
    desc_length = database.cols*8;

    //Create background classes (-1)
    vector<int> classes_bg = vector<int>(desc_bg.rows,-1);

    //Concatenate fg and bg classes
    classes = classes_bg;
    classes.insert(classes.end(), classes_fg.begin(), classes_fg.end());

    //Create descriptor matcher
    bfmatcher = DescriptorMatcher::create("BruteForce-Hamming");
}

void Matcher::matchGlobal(const vector<KeyPoint> & keypoints, const Mat descriptors,
        vector<Point2f> & points_matched, vector<int> & classes_matched)
{

    if (keypoints.size() == 0)
    {
        return;
    }

    vector<vector<DMatch> > matches;
    bfmatcher->knnMatch(descriptors, database, matches, 2);

    for (size_t i = 0; i < matches.size(); i++)
    {
        vector<DMatch> m = matches[i];

        float distance1 = m[0].distance / desc_length;
        float distance2 = m[1].distance / desc_length;
        int matched_class = classes[m[0].trainIdx];

        if (matched_class == -1) continue;
        if (distance1 > thr_dist) continue;
        if (distance1/distance2 > thr_ratio) continue;

        points_matched.push_back(keypoints[i].pt);
        classes_matched.push_back(matched_class);
    }
}

void Matcher::matchLocal(const vector<KeyPoint> & keypoints, const Mat descriptors,
        const Point2f center, const float scale, const float rotation,
        vector<Point2f> & points_matched, vector<int> & classes_matched)
{

    if (keypoints.size() == 0) {
        return;
    }

    //Transform initial points
    vector<Point2f> pts_fg_trans;
    pts_fg_trans.reserve(pts_fg_norm.size());
    for (size_t i = 0; i < pts_fg_norm.size(); i++)
    {
        pts_fg_trans.push_back(scale * rotate(pts_fg_norm[i], -rotation));
    }

    //Perform local matching
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        //Normalize keypoint with respect to center
        Point2f location_rel = keypoints[i].pt - center;

        //Find potential indices for matching
        vector<int> indices_potential;
        for (size_t j = 0; j < pts_fg_trans.size(); j++)
        {
            float l2norm = norm(pts_fg_trans[j] - location_rel);

            if (l2norm < thr_cutoff) {
                indices_potential.push_back(num_bg_points + j);
            }

        }

        //If there are no potential matches, continue
        if (indices_potential.size() == 0) continue;

        //Build descriptor matrix and classes from potential indices
        Mat database_potential = Mat(indices_potential.size(), database.cols, database.type());
        for (size_t j = 0; j < indices_potential.size(); j++) {
            database.row(indices_potential[j]).copyTo(database_potential.row(j));
        }

        //Find distances between descriptors
        vector<vector<DMatch> > matches;
        bfmatcher->knnMatch(descriptors.row(i), database_potential, matches, 2);

        vector<DMatch> m = matches[0];

        float distance1 = m[0].distance / desc_length;
        float distance2 = m.size() > 1 ? m[1].distance / desc_length : 1;

        if (distance1 > thr_dist) continue;
        if (distance1/distance2 > thr_ratio) continue;

        int matched_class = classes[indices_potential[m[0].trainIdx]];

        points_matched.push_back(keypoints[i].pt);
        classes_matched.push_back(matched_class);
    }
}

void Tracker::track(const Mat im_prev, const Mat im_gray, const vector<Point2f> & points_prev,
        vector<Point2f> & points_tracked, vector<unsigned char> & status)
{

    if (points_prev.size() > 0)
    {
        vector<float> err; //Needs to be float

        //Calculate forward optical flow for prev_location
        cv::calcOpticalFlowPyrLK(im_prev, im_gray, points_prev, points_tracked, status, err);

        vector<Point2f> points_back;
        vector<unsigned char> status_back;
        vector<float> err_back; //Needs to be float

        //Calculate backward optical flow for prev_location
        cv::calcOpticalFlowPyrLK(im_gray, im_prev, points_tracked, points_back, status_back, err_back);

        //Traverse vector backward so we can remove points on the fly
        for (int i = points_prev.size()-1; i >= 0; i--)
        {
            float l2norm = norm(points_back[i] - points_prev[i]);

            bool fb_err_is_large = l2norm > thr_fb;

            if (fb_err_is_large || !status[i] || !status_back[i])
            {
                points_tracked.erase(points_tracked.begin() + i);

                //Make sure the status flag is set to 0
                status[i] = 0;
            }

        }

    }
}
} /* namespace CMT */


#if 1 // standalone
int main(int argc, char** argv) {
    cv::VideoCapture cap(0);
    Mat frame;
    cap >> frame;

    // get bounding box
    Rect2d roi = cv::selectROI("tracker",frame,true,false);

    //quit if ROI was not selected
    if (roi.width == 0 || roi.height == 0)
        return 0;

    cv::Ptr<cv::Tracker> tracker = createTrackerCMT();
    tracker->init(frame, roi);

    // do the tracking
    printf("Start the tracking process, press ESC to quit.\n");
    for (;;) {
        // get frame from the video
        cap >> frame;

        // stop the program if no more images
        if (frame.rows == 0 || frame.cols == 0)
            break;

        // update the tracking result
        bool ok = tracker->update(frame, roi);
        if (ok) {
            // draw the tracked object
            cv::rectangle(frame, roi, cv::Scalar(255, 0, 0), 2, 1);
        } else {
            printf("lost !\n");
        }
        // show image with the tracked object
        cv::imshow("tracker", frame);

        //quit on ESC button
        if (cv::waitKey(1) == 27)break;
    }
}
#endif
