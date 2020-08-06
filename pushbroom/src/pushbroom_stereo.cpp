/**
 * Program that runs pushbroom stereo.
 * Copyright 2013-2015, Andrew Barry <abarry@csail.mit.edu>
 *
 */

#include "pushbroom_stereo.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <fstream>


using namespace cv;
using std::cout;
using std::endl;

namespace pushbroom {

namespace detail {

const int  NUMERIC_CONST=333; // just a constant that we multiply the score by to make
                          // all the parameters in a nice integer range
/**
 * Get the sum of absolute differences for a specific pixel location and disparity
 *
 * @param leftImage left imagename
 * @param rightImage right imagename
 * @param laplacianL laplacian-fitlered left image
 * @param laplacianR laplacian-filtered right image
 * @param pxX row pixel location
 * @param pxY column pixel location
 * @param state state structure that includes a number of parameters
 *
 * @retval scaled sum of absolute differences for this block --
 *      the value is the sum/numberOfPixels
 */
static int GetSAD(Mat leftImage, Mat rightImage, Mat laplacianL, Mat laplacianR, int pxX, int pxY, const StereoState &state)
{
    // init parameters
    int blockSize = state.blockSize;
    int disparity = state.disparity;
    int sobelLimit = state.sobelLimit;

    // top left corner of the SAD box
    int startX = pxX;
    int startY = pxY;

    // bottom right corner of the SAD box
    #ifndef USE_NEON
        int endX = pxX + blockSize - 1;
    #endif

    int endY = pxY + blockSize - 1;

    int leftVal = 0, rightVal = 0;

    int sad = 0;

    #ifdef USE_NEON
        uint16x8_t interest_op_sum_8x_L, interest_op_sum_8x_R, sad_sum_8x;

        // load zeros into everything
        interest_op_sum_8x_L = vdupq_n_u16(0);
        interest_op_sum_8x_R = vdupq_n_u16(0);
        sad_sum_8x = vdupq_n_u16(0);

    #endif

    for (int i=startY;i<=endY;i++) {
        // get a pointer for this row
        uchar *this_rowL = leftImage.ptr<uchar>(i);
        uchar *this_rowR = rightImage.ptr<uchar>(i);

        uchar *this_row_laplacianL = laplacianL.ptr<uchar>(i);
        uchar *this_row_laplacianR = laplacianR.ptr<uchar>(i);

        #ifdef USE_NEON
            // load this row into memory
            uint8x8_t this_row_8x8_L = vld1_u8(this_rowL + startX);
            uint8x8_t this_row_8x8_R = vld1_u8(this_rowR + startX + disparity);

            uint8x8_t interest_op_8x8_L = vld1_u8(this_row_laplacianL + startX);
            uint8x8_t interest_op_8x8_R = vld1_u8(this_row_laplacianR + startX + disparity);

            // do absolute differencing for the entire row in one operation!
            uint8x8_t sad_8x = vabd_u8(this_row_8x8_L, this_row_8x8_R);

            // sum up
            sad_sum_8x = vaddw_u8(sad_sum_8x, sad_8x);

            // sum laplacian values
            interest_op_sum_8x_L = vaddw_u8(interest_op_sum_8x_L, interest_op_8x8_L);
            interest_op_sum_8x_R = vaddw_u8(interest_op_sum_8x_R, interest_op_8x8_R);

        #else // USE_NEON

            for (int j=startX;j<=endX;j++) {
                // we are now looking at a single pixel value

                uchar sL = this_row_laplacianL[j];
                uchar sR = this_row_laplacianR[j + disparity];

                leftVal += sL;
                rightVal += sR;

                uchar pxL = this_rowL[j];
                uchar pxR = this_rowR[j + disparity];

                sad += abs(pxL - pxR);
            }
        #endif // USE_NEON
    }

    #ifdef USE_NEON
        // sum up
        sad = vgetq_lane_u16(sad_sum_8x, 0) + vgetq_lane_u16(sad_sum_8x, 1)
           + vgetq_lane_u16(sad_sum_8x, 2) + vgetq_lane_u16(sad_sum_8x, 3)
           + vgetq_lane_u16(sad_sum_8x, 4);// + vgetq_lane_u16(sad_sum_8x, 5)
    //           + vgetq_lane_u16(sad_sum_8x, 6) + vgetq_lane_u16(sad_sum_8x, 7);

        leftVal = vgetq_lane_u16(interest_op_sum_8x_L, 0)
                + vgetq_lane_u16(interest_op_sum_8x_L, 1)
                + vgetq_lane_u16(interest_op_sum_8x_L, 2)
                + vgetq_lane_u16(interest_op_sum_8x_L, 3)
                + vgetq_lane_u16(interest_op_sum_8x_L, 4);


        rightVal = vgetq_lane_u16(interest_op_sum_8x_R, 0)
                 + vgetq_lane_u16(interest_op_sum_8x_R, 1)
                 + vgetq_lane_u16(interest_op_sum_8x_R, 2)
                 + vgetq_lane_u16(interest_op_sum_8x_R, 3)
                 + vgetq_lane_u16(interest_op_sum_8x_R, 4);
    #endif


    if (leftVal < sobelLimit || rightVal < sobelLimit)
    {
        return -1;
    }

    // weight laplacian_value into the score
    int laplacian_value = leftVal + rightVal;
    return NUMERIC_CONST*(float)sad/(float)laplacian_value;
}

/**
 * Checks for horizontal invariance by searching near the zero-disparity region
 * for good matches.  If we find a match, that indicates that this is likely not
 * a true obstacle since it matches in more places than just the single-disparity
 * check.
 *
 * @param leftImage left image
 * @param rightImage right image
 * @param pxX column pixel location
 * @param pxY row pixel location
 * @param state state structure that includes a number of parameters
 *
 * @retval true if there is another match (so NOT an obstacle)
 */
static bool CheckHorizontalInvariance(const Mat &leftImage, const Mat &rightImage, const Mat &sobelL,
    const Mat &sobelR, int pxX, int pxY, const StereoState &state) {

    const int  INVARIANCE_CHECK_VERT_OFFSET_MIN=-8;
    const int  INVARIANCE_CHECK_VERT_OFFSET_MAX=8;
    const int  INVARIANCE_CHECK_VERT_OFFSET_INCREMENT=2;
    const int  INVARIANCE_CHECK_HORZ_OFFSET_MIN=-3;
    const int  INVARIANCE_CHECK_HORZ_OFFSET_MAX=3;

    // init parameters
    int blockSize = state.blockSize;
    int disparity = state.zero_dist_disparity;
    int sobelLimit = state.sobelLimit;

    // top left corner of the SAD box
    int startX = pxX;
    int startY = pxY;

    // bottom right corner of the SAD box
    int endX = pxX + blockSize - 1;
    int endY = pxY + blockSize - 1;

    // if we are off the edge of the image so we can't tell if this
    // might be an issue -- give up and return true
    // (note: this used to be false and caused bad detections on real flight
    // data near the edge of the frame)
    if (   startX + disparity + INVARIANCE_CHECK_HORZ_OFFSET_MIN < 0
        || endX + disparity + INVARIANCE_CHECK_HORZ_OFFSET_MAX > rightImage.cols) {

        return true;
    }

    if (startY + INVARIANCE_CHECK_VERT_OFFSET_MIN < 0
        || endY + INVARIANCE_CHECK_VERT_OFFSET_MAX > rightImage.rows) {
        // we are limited in the vertical range we can check here

        // TODO: be smarter here

        // give up and bail out, deleting potential hits
        return true;

    }


    // here we check a few spots:
    //  1) the expected match at zero-disparity (10-infinity meters away)
    //  2) inf distance, moved up 1-2 pixels
    //  3) inf distance, moved down 1-2 pixels
    //  4) others?

    // first check zero-disparity
    int leftVal = 0;

    int right_val_array[400];
    int sad_array[400];
    int sobel_array[400];

    for (int i=0;i<400;i++) {
        right_val_array[i] = 0;
        sad_array[i] = 0;
        sobel_array[i] = 0;
    }

    int counter = 0;

    for (int i=startY;i<=endY;i++)
    {
        for (int j=startX;j<=endX;j++)
        {
            // we are now looking at a single pixel value
            uchar pxL = leftImage.at<uchar>(i,j);

            uchar pxR_array[400], sR_array[400];

            // for each pixel in the left image, we are going to search a bunch
            // of pixels in the right image.  We do it this way to save the computation
            // of dealing with the same left-image pixel over and over again.

            // counter indexes which location we're looking at for this run, so for each
            // pixel in the left image, we examine a bunch of pixels in the right image
            // and add up their results into different slots in sad_array over the loop
            counter = 0;

            for (int vert_offset = INVARIANCE_CHECK_VERT_OFFSET_MIN;
                vert_offset <= INVARIANCE_CHECK_VERT_OFFSET_MAX;
                vert_offset+= INVARIANCE_CHECK_VERT_OFFSET_INCREMENT) {

                for (int horz_offset = INVARIANCE_CHECK_HORZ_OFFSET_MIN;
                    horz_offset <= INVARIANCE_CHECK_HORZ_OFFSET_MAX;
                    horz_offset++) {

                    pxR_array[counter] = rightImage.at<uchar>(i + vert_offset, j + disparity + horz_offset);
                    sR_array[counter] = sobelR.at<uchar>(i + vert_offset, j + disparity + horz_offset);
                    right_val_array[counter] += sR_array[counter];

                    sad_array[counter] += abs(pxL - pxR_array[counter]);

                    counter ++;
                }
            }

            uchar sL = sobelL.at<uchar>(i,j);

            leftVal += sL;

        }
    }

    for (int i = 0; i < counter; i++)
    {
        sobel_array[i] = leftVal + right_val_array[i];

        // we don't check for leftVal >= sobelLimit because we have already
        // checked that in the main search loop (in GetSAD).
        //if (right_val_array[i] >= sobelLimit && 100*(float)sad_array[i]/(float)((float)sobel_array[i]*state.interestOperatorMultiplierHorizontalInvariance) < state.sadThreshold) {
        if (right_val_array[i] >= sobelLimit && NUMERIC_CONST*state.horizontalInvarianceMultiplier*(float)sad_array[i]/((float)sobel_array[i]) < state.sadThreshold) {
            return true;
        }
    }
    return false;
}

static void RunStereo(
    const Mat &leftImage, const Mat &rightImage, const Mat &laplacian_left, const Mat &laplacian_right,
    std::vector<Point3f> &pointVector3d, std::vector<Point3i> &pointVector2d, std::vector<uchar> &pointColors,
    const StereoState &state
     )
{
    // we will do this by looping through every block in the left image
    // (defined by blockSize) and checking for a matching value on
    // the right image

    std::vector<Point3f> localHitPoints;

    int blockSize = state.blockSize;
    int disparity = state.disparity;
    int sadThreshold = state.sadThreshold;

    int startJ = 0;
    int stopJ = leftImage.cols - (disparity + blockSize);
    if (disparity < 0)
    {
        startJ = -disparity;
        stopJ = leftImage.cols - blockSize;
    }

    int hitCounter = 0;
    int row_start = 0;
    int row_end = leftImage.rows-blockSize;

    for (int i=row_start; i < row_end; i+=blockSize)
    {
        for (int j=startJ; j < stopJ; j+=blockSize)
        {
            // get the sum of absolute differences for this location on both images
            int sad = GetSAD(leftImage, rightImage, laplacian_left, laplacian_right, j, i, state);
            // if the SAD is below the threshold, we got a hit
            if (sad < sadThreshold && sad >= 0)
            {
                if (! CheckHorizontalInvariance(leftImage, rightImage, laplacian_left, laplacian_right, j, i, state))
                    continue;

                // centered version
                Point c(j+blockSize/2.0, i+blockSize/2.0);
                localHitPoints.push_back(Point3f(c.x, c.y, -disparity));

                uchar pxL = leftImage.at<uchar>(c);
                pointColors.push_back(pxL);

                // sad saved as artificial 3d coord
                pointVector2d.push_back(Point3i(c.x, c.y, sad));
                hitCounter ++;
            }
        }
    }
    // now we have an array of hits -- transform them to 3d points
    if (hitCounter > 0) {
        perspectiveTransform(localHitPoints, pointVector3d, state.Q);
    }
}

}; // namespace detail

//
//
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
void Stereo(InputArray _leftImage, InputArray _rightImage, std::vector<Point3f> &pointVector3d, std::vector<uchar> &pointColors, std::vector<Point3i> &pointVector2d, const StereoState &state) {

    Mat leftImage = _leftImage.getMat();
    Mat rightImage = _rightImage.getMat();

    // make sure that the inputs are of the right type
    CV_Assert(leftImage.type() == CV_8UC1 && rightImage.type() == CV_8UC1);

    // remap (undistort)
    Mat remapped_left(state.mapxL.rows, state.mapxL.cols, leftImage.depth());
    Mat remapped_right(state.mapxR.rows, state.mapxR.cols, rightImage.depth());

    remap(leftImage, remapped_left, state.mapxL, Mat(), INTER_NEAREST);
    remap(rightImage, remapped_right, state.mapxR, Mat(), INTER_NEAREST);

    // apply interest operator
    Mat laplacian_left(remapped_left.rows, remapped_left.cols, remapped_left.depth());
    Mat laplacian_right(remapped_right.rows, remapped_right.cols, remapped_right.depth());

    Laplacian(remapped_left, laplacian_left, -1, 3, 1, 0, BORDER_DEFAULT);
    Laplacian(remapped_right, laplacian_right, -1, 3, 1, 0, BORDER_DEFAULT);

    detail::RunStereo(
        remapped_left,remapped_right, laplacian_left,laplacian_right,
        pointVector3d, pointVector2d, pointColors,
        state);    //cout << "[main] imshow2 ok" <   < endl;

} // Stereo

}; // namespace pushbroom

