

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/tracking.hpp>

using namespace cv::ml;
using namespace cv;
using namespace std;

static Mat RLTikh_deconvolution(Mat observed, Mat psf, int iterations) {

    Mat deconv = observed.clone();
    double mu = 0.01;

    // Iterate
    for (int i = 0; i < iterations; i++) {

            // Temporary matrix
            Mat ratio;
            sepFilter2D(deconv, ratio, deconv.depth(), psf, psf, Point(-1, -1), 0,
            BORDER_REFLECT);

            divide(observed, ratio, ratio);

            sepFilter2D(ratio, ratio, ratio.depth(), psf, psf, Point(-1, -1), 0,
            BORDER_REFLECT);

            // TV Regularization
            Mat denom;
            Laplacian(deconv, denom, deconv.depth(), 1, 1, 0, BORDER_REFLECT);
            denom = 1.0 - 2.0 * mu * denom;
            divide(ratio, denom, ratio);

            // Apply iteration on the estimate
            multiply(deconv, ratio, deconv);
    }

    return deconv;
}
static Mat RLTikh_deconvolution2(Mat observed, Mat psf, int iterations) {

    Mat deconv = observed.clone();
    double mu = 0.01;

    // Iterate
    for (int i = 0; i < iterations; i++) {

            // Temporary matrix
            Mat ratio;
            filter2D(deconv, ratio, deconv.depth(), psf, Point(-1, -1), 0,
            BORDER_REFLECT);

            divide(observed, ratio, ratio);

            filter2D(ratio, ratio, ratio.depth(), psf, Point(-1, -1), 0,
            BORDER_REFLECT);

            // TV Regularization
            Mat denom;
            Laplacian(deconv, denom, deconv.depth(), 1, 1, 0, BORDER_REFLECT);
            denom = 1.0 - 2.0 * mu * denom;
            divide(ratio, denom, ratio);

            // Apply iteration on the estimate
            multiply(deconv, ratio, deconv);
    }

    return deconv;
}



int main()
{
    Mat img = imread("im1.jpg");
    img.convertTo(img, CV_32F);
    Mat psf = getGaussianKernel(13, -1);
    Mat out = RLTikh_deconvolution(img,psf,5);
    imshow("out",out);
    Mat psf2 = psf.t() * psf;
    cv::sqrt(psf2,psf2);
    Mat out2 = RLTikh_deconvolution2(img,psf2,5);
    imshow("out2",out2);
    waitKey();
    return 0;
}
