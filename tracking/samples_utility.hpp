#ifndef _SAMPLES_UTILITY_HPP_
#define _SAMPLES_UTILITY_HPP_

#include <opencv2/tracking.hpp>

extern cv::Ptr<cv::Tracker> createTrackerCT();
extern cv::Ptr<cv::Tracker> createTrackerCMT();
extern cv::Ptr<cv::Tracker> createTrackerBACF();

inline cv::Ptr<cv::Tracker> createTrackerByName(cv::String name)
{
    cv::Ptr<cv::Tracker> tracker;

    if (name == "KCF")
        tracker = cv::TrackerKCF::create();
    else if (name == "TLD")
        tracker = cv::TrackerTLD::create();
    else if (name == "BOOSTING")
        tracker = cv::TrackerBoosting::create();
    else if (name == "MEDIAN_FLOW")
        tracker = cv::TrackerMedianFlow::create();
    else if (name == "MIL")
        tracker = cv::TrackerMIL::create();
    else if (name == "GOTURN")
        tracker = cv::TrackerGOTURN::create();
    else if (name == "MOSSE")
        tracker = cv::TrackerMOSSE::create();
//    else if (name == "SRDCF")
//        tracker = cv::TrackerSRDCF::create();
    else if (name == "CSRT")
        tracker = cv::TrackerCSRT::create();
    else if (name == "CMT")
        tracker = createTrackerCMT();
    else if (name == "CT")
        tracker = createTrackerCT();
    else if (name == "BACF")
        tracker = createTrackerBACF();
    else
        CV_Error(cv::Error::StsBadArg, "Invalid tracking algorithm name\n");

    return tracker;
}

#endif
