#ifndef __push_broom_stereo_onboard__
#define __push_broom_stereo_onboard__
#include "opencv2/core.hpp"

namespace pushbroom {

struct StereoState
{
    // pushbroom params
    int disparity;
    int zero_dist_disparity;
    int sobelLimit;
    int blockSize;
    int sadThreshold;
    float horizontalInvarianceMultiplier;

    // serialized results from stereo calibration
    cv::Mat mapxL;
    cv::Mat mapxR;
    cv::Mat Q;
};


/**
 * Sum-of-absolute-differences (SAD) algorithm on a single disparity
 * This is the public interface
 *
 * @param leftImage left image
 * @param rightImage right image
 * @param pointVector3d 3d output, projected potential obstacle
 * @param pointVector2d 3d output, (x,y,sad)
 * @param state state structure that includes a number of parameters
*/
void Stereo(cv::InputArray _leftImage, cv::InputArray _rightImage, std::vector<cv::Point3f> &pointVector3d, std::vector<uchar> &pointColors, std::vector<cv::Point3i> &pointVector2d, const StereoState &state);

}

#endif // __push_broom_stereo_onboard__