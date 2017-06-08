#ifndef __mace_h_onboard__
#define __mace_h_onboard__

#include "opencv2/core.hpp"


//
//! Minimum Average Correlation Energy Filter
//!   useful for authentification with (cancellable) biometrical features.
//!   (does not need many positives to train (~10), and no negatives at all, also robust to noise/salting)
//  algorithm largely taken from : https://github.com/polyu/faceservergpl
//

struct MACE {
    //! train it on positive features
    virtual void compute(const std::vector<cv::Mat> &images) = 0;

    //! predict correlation values on a query image
    //!   return [0..1] based value is: (100 * pcpe * pslr)
    virtual double correlate(const cv::Mat &img) = 0;

	//
	//! images will get resized to SIZE_OF_IMAGE
	//! if salt is != 0, a random convolution, seeded by salt, will get applied to all images
	//!   to achieve "cancellable features".
	//
	static cv::Ptr<MACE> create(int SIZE_OF_IMAGE, int salt=0);

};

#endif // __mace_h_onboard__
