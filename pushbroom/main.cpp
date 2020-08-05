/**
 * Program that runs pushbroom stereo.
 * Copyright 2013-2015, Andrew Barry <abarry@csail.mit.edu>
 *
 */

//#include "pushbroom-stereo.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <fstream>


using namespace cv;
using std::cout;
using std::endl;

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
int GetSAD(Mat leftImage, Mat rightImage, Mat laplacianL, Mat laplacianR, int pxX, int pxY, const StereoState &state)
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
bool CheckHorizontalInvariance(const Mat &leftImage, const Mat &rightImage, const Mat &sobelL,
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

void RunStereo(
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
        remapped_left,remapped_right,laplacian_left,laplacian_right,
        pointVector3d, pointVector2d, pointColors,
        state);    //cout << "[main] imshow2 ok" <   < endl;
}
}; // namespace pushbroom


Mat ident(int h, int w) {
    Mat_<Point2f> m(h,w);
    for (int i=0; i<h; i++)
        for (int j=0; j<w; j++)
            m(i,j)=Point2f(j,i);
    return m;
}

int main() {
    // initilize default parameters
    pushbroom::StereoState state;

    // https://github.com/andybarry/flight/blob/master/sensors/stereo/deltawing-stereo-odroid-cam1.conf
    state.disparity = -33;
    state.zero_dist_disparity = -21;
    state.sobelLimit = 860;
    state.horizontalInvarianceMultiplier = 0.5;
    state.blockSize = 5;
    state.sadThreshold = 54;


    if (state.blockSize > 10 || state.blockSize < 1)
    {
        fprintf(stderr, "Warning: block size is very large "
            "or small (%d).  Expect trouble.\n", state.blockSize);
    }
    if (0) {
        auto read_map = [](int k, Mat &dst) {
            Mat mx,my;
            FileStorage MX(format("mx%d.xml",0).c_str(), 0);
            MX[format("mx%d",k)] >> mx;
            FileStorage MY(format("my%d.xml",0).c_str(), 0);
            MY[format("my%d",k)] >> my;
            std::vector<Mat> mv {mx,my};
            merge(mv, dst);
        };
        read_map(1, state.mapxL);
        read_map(2, state.mapxR);
    } else {
        state.mapxL = ident(240,376);
        state.mapxR = ident(240,376);
    }
    FileStorage q("Q.xml",0);
    q["Q"] >> state.Q;

    while(1) {

        // capture images from the cameras
        Mat matL, matR;
        matL = imread("ambush_5_left.jpg", 0);
        matR = imread("ambush_5_right.jpg", 0);
        resize(matL,matL,state.mapxL.size());
        resize(matR,matR,state.mapxR.size());

        std::vector<Point3f> pointVector3d;
        std::vector<uchar> pointColors;
        std::vector<Point3i> pointVector2d; // for display

        int64 t0 = getTickCount();
        //pushbroom_stereo.ProcessImages(matL, matR, &pointVector3d, &pointColors, &pointVector2d, state);
        pushbroom::Stereo(matL, matR, pointVector3d, pointColors, pointVector2d, state);
        int64 t1 = getTickCount();

        Mat remapL;
        remap(matL, remapL, state.mapxL, Mat(), INTER_NEAREST);
        imshow("L",remapL);

        Mat remapR;
        remap(matR, remapR, state.mapxR, Mat(), INTER_NEAREST);
        imshow("R",remapR);

        //https://github.com/andybarry/flight/blob/master/sensors/bm-stereo/bm-stereo.cpp#L112-L123
        Ptr<StereoBM> bm = StereoBM::create(3*16, 5);
        bm->setMinDisparity(0);
        bm->setSpeckleWindowSize(59);
        bm->setSpeckleRange(30);

        Mat disparity_bm, disp8;
        int64 t2 = getTickCount();
        bm->compute(remapL, remapR, disparity_bm);
        int64 t3 = getTickCount();
        double T1 = (t1-t0) / getTickFrequency();
        double T2 = (t3-t2) / getTickFrequency();
        cout << "T " << T1 << " / " << T2 << endl;
        disparity_bm.convertTo(disp8, CV_8U);
        cvtColor(disp8,disp8,COLOR_GRAY2BGR);
        for (int i=0; i<(int)pointVector2d.size(); i++) {
            int x = pointVector2d.at(i).x;
            int y = pointVector2d.at(i).y;
            circle(disp8, Point(x, y), state.blockSize*2, Scalar(pointColors[i],10,pointColors[i]), cv::FILLED);
            circle(disp8, Point(x, y), state.blockSize, Scalar(20,20,5*pointColors[i]), cv::FILLED);
        }
        imshow("BM", disp8);

        Mat image_3d;
        reprojectImageTo3D(disparity_bm, image_3d, state.Q, true);
        //cout << "Q " << image_3d.size() << " " << image_3d.type() << endl;
        imshow("Q", image_3d);

        int i=0, off=state.blockSize/2;
        for (auto p:pointVector2d) {
            Point3f a = image_3d.at<Point3f>(p.y+off,p.x+off);
            Point3f b = pointVector3d[i];
            Point3i c = pointVector2d[i++];
            cout << a << " " << b << " " << norm(a-b) << "\t" << c << endl;
        }
        int k = waitKey();
        switch(k) {
            case 27: return 0;
            case 'D': state.disparity ++; cout << "disp " << state.disparity << endl; break;
            case 'd': state.disparity --; cout << "disp " << state.disparity << endl; break;
            case 'Z': state.zero_dist_disparity ++; cout << "zero " << state.zero_dist_disparity << endl; break;
            case 'z': state.zero_dist_disparity --; cout << "zero " << state.zero_dist_disparity << endl; break;
            case 'S': state.sadThreshold ++; cout << "sad  " << state.sadThreshold << endl; break;
            case 's': state.sadThreshold --; cout << "sad  " << state.sadThreshold << endl; break;
            case 'B': state.blockSize ++; cout << "block " << state.blockSize << endl; break;
            case 'b': state.blockSize --; cout << "block " << state.blockSize << endl; break;
        }
    }
    return 0;
}
