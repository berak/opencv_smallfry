#ifndef __lbptop_onboard__

#include <opencv2/opencv.hpp>
#include <deque>


typedef std::deque< cv::Mat_<uchar> > Sequence;

/*
void lbp_xy(const Sequence &seq, cv::Mat &hist, cv::Point p);
void lbp_xz(const Sequence &seq, cv::Mat &hist, cv::Point p);
void lbp_yz(const Sequence &seq, cv::Mat &hist, cv::Point p);
*/

void lbp_set_blocksize(int s);
int  lbp_get_blocksize();

cv::Mat lbptop(const Sequence &seq, const std::vector<cv::Point> &samps);

cv::Mat img_yz(const Sequence &seq, cv::Point p);
cv::Mat img_xz(const Sequence &seq, cv::Point p);


#endif // __lbptop_onboard__
