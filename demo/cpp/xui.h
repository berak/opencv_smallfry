
#ifndef __opencv_gui_onboard__
#define __opencv_gui_onboard__

#include <opencv2/opencv.hpp>
#include <vector>

struct UI
{
	//! a 2 state button
    virtual void addButton(cv::String name, const cv::Rect &r, cv::Scalar c, int &v) = 0;
    //! a slider with  r.width values
    virtual void addSlider(cv::String name, const cv::Rect &r, cv::Scalar c, int &v) = 0;
    //! a 2 state button, only sets to high, resetting to low is left to user.
    virtual void addTrigger(cv::String name, const cv::Rect &r, cv::Scalar c, int &v) = 0;
    //! selection of a string list, click left for prev, right for next
    virtual void addChoice(std::vector<cv::String> &choice, const cv::Rect &r, cv::Scalar c, int &v) = 0;
    virtual void setText(int i, cv::String t) = 0;
    virtual cv::String info() = 0;
    virtual bool toggle() = 0;
    virtual void show(cv::Mat &img) = 0;
};

cv::Ptr<UI> createUI(cv::String n);

#endif // __opencv_gui_onboard__

