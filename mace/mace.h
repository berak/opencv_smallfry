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

    //! optionally encrypt images with random convolution
    // @param int salz : the seed for the random convolution
    virtual void salt(int salz) = 0;

    //! train it on positive features,
    //!   compute the mace filter: `h = D(-1) * X * (X(+) * D(-1) * X)(-1) * C`
    //!   also calculate a minimal threshold for this class, the smallest self-similarity from the  train images
    // @param images : a vector<Mat>
    virtual void train(cv::InputArrayOfArrays images) = 0;

    //! correlate query img and threshold to min class value
    // @param query : a Mat with query image
    virtual bool same(cv::InputArray query) const = 0;


    // @param IMGSIZE : images will get resized to this
    static cv::Ptr<MACE> create(int IMGSIZE);

    //! multiple filters applied to subregions of the image.
    // @param IMGSIZE : images will get resized to this internally
    // @param rects : vector<Rect2f> in [0..1] coords
    static cv::Ptr<MACE> createSampler(int IMGSIZE, cv::InputArray rects);
};

#endif // __mace_h_onboard__
