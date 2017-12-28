#ifndef __mace_h_onboard__
#define __mace_h_onboard__

#include "opencv2/core.hpp"


//
//! Minimum Average Correlation Energy Filter
//!   useful for authentification with (cancellable) biometrical features.
//!   (does not need many positives to train (~10), and no negatives at all, also robust to noise/salting)
//    s.a: "CANCELABLE BIOMETRIC FILTERS FOR FACE RECOGNITION", Marios Savvides, B.V.K. Vijaya Kumar and P.K. Khosla
//

struct MACE : cv::Algorithm {

    //! optionally encrypt images with random convolution
    // @param int seed : the seed for the random convolution
    virtual void salt(int64 seed) = 0;
    //! optionally encrypt images with random convolution
    // @param passphrase: generates a crc64 seed for the random convolution
    virtual void salt(const cv::String &passphrase) = 0;

    //! train it on positive features,
    //!   compute the mace filter: `h = D(-1) * X * (X(+) * D(-1) * X)(-1) * C`
    //!   also calculate a minimal threshold for this class, the smallest self-similarity from the train images
    // @param images : a vector<Mat>
    virtual void train(cv::InputArrayOfArrays images) = 0;

    //! correlate query img and threshold to min class value
    // @param query : a Mat with query image
    virtual bool same(cv::InputArray query) const = 0;


    // @param IMGSIZE : images will get resized to this (should be even)
    static cv::Ptr<MACE> create(int IMGSIZE=64);
};

#endif // __mace_h_onboard__
