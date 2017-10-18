#ifndef __lbptop_onboard__

#include <opencv2/opencv.hpp>
#include <deque>


typedef std::deque< cv::Mat_<uchar> > Sequence;

/*
void lbp_xy(const Sequence &seq, cv::Mat &hist, cv::Point p);
void lbp_xz(const Sequence &seq, cv::Mat &hist, cv::Point p);
void lbp_yz(const Sequence &seq, cv::Mat &hist, cv::Point p);
*/

//void lbp_set_blocksize(int s);
//int  lbp_get_blocksize();

cv::Mat lbptop(const Sequence &seq);

cv::Mat img_yz(const Sequence &seq, const cv::Rect &r);
cv::Mat img_xz(const Sequence &seq, const cv::Rect &r);

int lbpFlow(const cv::String &filename, cv::Mat &desc, int frameFrom=0, int frameTo=0);


#endif // __lbptop_onboard__
