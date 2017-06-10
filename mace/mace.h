#ifndef __mace_h_onboard__
#define __mace_h_onboard__

#include "opencv2/core.hpp"


//
//! Minimum Average Correlation Energy Filter
//!   useful for authentification with (cancellable) biometrical features.
//!   (does not need many positives to train (~10), and no negatives at all, also robust to noise/salting)
//  algorithm largely taken from : https://github.com/polyu/faceservergpl
//

struct MACE : cv::Algorithm {

    //! optionally encrypt images with seeded random convolution
    virtual void salt(int salz) = 0;

    //! train it on positive features (e.g. a vector<Mat>)
    virtual void train(cv::InputArrayOfArrays images) = 0;

    //! correlate query img and compare to min positive value
    virtual bool same(cv::InputArray query) const = 0;

    //! images will get resized to IMGSIZE
    static cv::Ptr<MACE> create(int IMGSIZE);

    //! multiple filters applied to subregions given in [0..1] rects
    static cv::Ptr<MACE> createSampler(int IMGSIZE, cv::InputArray rects);
};

#endif // __mace_h_onboard__
