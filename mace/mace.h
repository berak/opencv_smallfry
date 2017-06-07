#ifndef __mace_h_onboard__
#define __mace_h_onboard__

#include "opencv2/core.hpp"


//
//! Minimum Average Correlation Energy Filter
//!   useful for authentification with (cancellable) biometrical features.
//!   (does not need many positives to train (~10), and no negatives at all, also robust to noise/salting)
// algorithm largely taken from : https://github.com/polyu/faceservergpl
//

struct MACE {
    //! train it on positive features
    virtual void compute(const std::vector<cv::Mat> &images) = 0;

    //! predict correlation values on a query image
    //! (100 * pcpe * pslr) makes a nice correlation value
    virtual void correlate(const cv::Mat &img, double &peakCorrPlaneEnergy, double &peakToSideLobeRatio) = 0;
};
cv::Ptr<MACE> createMACE(int SIZE_OF_IMAGE);


#endif // __mace_h_onboard__
