#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "mace.h"
#include "profile.h"

using namespace cv;

// Rearrange the quadrants of Fourier image so that the origin is at
// the image center
// src & dst arrays of equal size & type
void shiftDFT(const Mat &src, Mat &dst) {
    PROFILE
    Size size = src.size();

    if (dst.empty() || (dst.size().width != size.width || dst.size().height != size.height)) {
        dst.create(src.size(), src.type());
    }

    int cx = size.width/2;
    int cy = size.height/2; // image center

    Mat q1 = src(Rect(0, 0, cx,cy));
    Mat q2 = src(Rect(cx,0, cx,cy));
    Mat q3 = src(Rect(cx,cy,cx,cy));
    Mat q4 = src(Rect(0, cy,cx,cy));
    Mat d1 = dst(Rect(0, 0, cx,cy));
    Mat d2 = dst(Rect(cx,0, cx,cy));
    Mat d3 = dst(Rect(cx,cy,cx,cy));
    Mat d4 = dst(Rect(0, cy,cx,cy));

    if (src.data != dst.data){
        q3.copyTo(d1);
        q4.copyTo(d2);
        q1.copyTo(d3);
        q2.copyTo(d4);
    } else {
        Mat tmp;
        q3.copyTo(tmp);
        q1.copyTo(q3);
        tmp.copyTo(q1);
        q4.copyTo(tmp);
        q2.copyTo(q4);
        tmp.copyTo(q2);
    }
}

Mat dftImage(Mat img, int SIZE_OF_IMAGE) {
    PROFILE;
    Mat gray;
    resize(img, gray, Size(SIZE_OF_IMAGE,SIZE_OF_IMAGE)) ;
    equalizeHist(gray,gray);

    Mat input[2];
    gray.convertTo(input[0], CV_64F);
    input[1] = Mat(input[0].size(), input[0].type(), 0.0);
    Mat complexInput; merge(input, 2, complexInput);

    Mat_<Vec2d> dftImg(SIZE_OF_IMAGE*2, SIZE_OF_IMAGE*2, 0.0);
    complexInput.copyTo(dftImg(Rect(0,0,SIZE_OF_IMAGE,SIZE_OF_IMAGE)));

    dft(dftImg, dftImg);
    return dftImg;
}



struct MACEImpl : MACE {
    int SIZE_OF_IMAGE;
    Mat_<Vec2d> maceFilterVisualize; // filled from compute()

    MACEImpl(int s) : SIZE_OF_IMAGE(s) {}

    void compute(const std::vector<Mat> &img) {
        PROFILE
        int size = img.size();
        int SIZE_OF_IMAGE_2X = SIZE_OF_IMAGE * 2;
        int TOTALPIXEL = SIZE_OF_IMAGE_2X * SIZE_OF_IMAGE_2X;

        Mat_<Vec2d> D(TOTALPIXEL, 1, 0.0);
        Mat_<Vec2d> S(TOTALPIXEL, size, 0.0);
        Mat_<Vec2d> SPLUS(size, TOTALPIXEL, 0.0);

        for (size_t i=0; i<size; i++) {
            Mat_<Vec2d> dftImg = dftImage(img[i], SIZE_OF_IMAGE);
            for (int l=0; l<SIZE_OF_IMAGE_2X; l++) {
                for (int m=0; m<SIZE_OF_IMAGE_2X; m++) {
                    int j = l * SIZE_OF_IMAGE_2X + m;
                    Vec2d s = dftImg(l, m);
                    S(j, i) = s;
                    SPLUS(i, j) = Vec2d(s[0], -s[1]);

                    double val=((pow(s[0],2) + pow(s[1],2)));
                    D(j, 0)[0] += val;
                }
            }
        }

        Mat_<Vec2d> DINV(TOTALPIXEL, 1, 0.0);
        for (int i=0; i<TOTALPIXEL; i++) {
            DINV(i,0) = Vec2d((SIZE_OF_IMAGE_2X*SIZE_OF_IMAGE_2X*size)/sqrt(D(i,0)[0]), 0);
        }

        Mat_<Vec2d> DINV_S(TOTALPIXEL, size, 0.0);
        Mat_<Vec2d> SPLUS_DINV(size, TOTALPIXEL, 0.0);
        for (int l=0; l<size; l++) {
            for (int m=0; m<TOTALPIXEL; m++) {
                SPLUS_DINV(l, m)[0] = SPLUS(l,m)[0] * DINV(m,0)[0];
                SPLUS_DINV(l, m)[1] = SPLUS(l,m)[1] * DINV(m,0)[1];
                DINV_S(m, l)[0] = S(m,l)[0] * DINV(m,0)[0];
                DINV_S(m, l)[1] = S(m,l)[1] * DINV(m,0)[1];
            }
        }

        Mat_<Vec2d> SPLUS_DINV_S = SPLUS_DINV * S;
        Mat_<Vec2d> SPLUS_DINV_S_INV(size, size);
        Mat_<double> SPLUS_DINV_S_INV_1(2*size, 2*size);
        for (size_t l=0; l<size; l++) {
            for (size_t m=0; m<size; m++) {
                Vec2d s = SPLUS_DINV_S(l, m);
                SPLUS_DINV_S_INV_1(l,      m)      = s[0];
                SPLUS_DINV_S_INV_1(l+size, m+size) = s[0];
                SPLUS_DINV_S_INV_1(l,      m+size) = s[1];
                SPLUS_DINV_S_INV_1(l+size, m)     = -s[1];
            }
        }
        invert(SPLUS_DINV_S_INV_1, SPLUS_DINV_S_INV_1);

        for (int l=0; l<size; l++) {
            for (int m=0; m<size; m++) {
                SPLUS_DINV_S_INV(l, m) = Vec2d(SPLUS_DINV_S_INV_1(l,m), SPLUS_DINV_S_INV_1(l,m+size));
            }
        }

        Mat Hmace = DINV_S * SPLUS_DINV_S_INV;
        Mat C(size,1,CV_64FC2, Scalar(1,0));
        Mat_<Vec2d> Hmace_FIN = Hmace * C;
        maceFilterVisualize.create(SIZE_OF_IMAGE_2X, SIZE_OF_IMAGE_2X);
        for (int l=0; l<SIZE_OF_IMAGE_2X; l++) {
            for (int m=0; m<SIZE_OF_IMAGE_2X; m++) {
                maceFilterVisualize(l, m) = Hmace_FIN(l * SIZE_OF_IMAGE_2X + m, 0);
            }
        }
    }

    void correlate(const Mat &img, double &peakCorrPlaneEnergy, double &peakToSideLobeRatio)
    {
        PROFILE
        CV_Assert(! maceFilterVisualize.empty()); // not trained.

        int  SIZE_OF_IMAGE_2X = SIZE_OF_IMAGE * 2;
        Mat dftImg = dftImage(img, SIZE_OF_IMAGE);
        mulSpectrums(dftImg , maceFilterVisualize, dftImg, CV_DXT_MUL_CONJ);
        dft(dftImg, dftImg, CV_DXT_INV_SCALE, 0);
        Mat chn[2];
        split(dftImg, chn);
        Mat_<double> re;
        shiftDFT(chn[0], re);
        double m1,M1;
        minMaxLoc(re, &m1, &M1, 0, 0);
        peakCorrPlaneEnergy = M1 / sqrt(sum(re)[0]);

        re -= m1;
#if 0
imshow("RE",re*10000);
waitKey();
#endif
        double value=0;
        double num=0;
        int rad1=int(floor((double)(45.0/64.0)*(double)SIZE_OF_IMAGE));
        int rad2=int(floor((double)(27.0/64.0)*(double)SIZE_OF_IMAGE));

        for (int l=0; l<SIZE_OF_IMAGE_2X; l++) {
            for (int m=0; m<SIZE_OF_IMAGE_2X; m++) {
                double rad=sqrt((pow(m-SIZE_OF_IMAGE,2) + pow(l-SIZE_OF_IMAGE,2)));
                if (rad < rad1) {
                    if (rad > rad2) {
                        value += re(l,m);
                        num++;
                    }
                }
            }
        }
        value /= num;

        double std2=0;
        for (int l=0; l<SIZE_OF_IMAGE_2X; l++) {
            for (int m=0; m<SIZE_OF_IMAGE_2X; m++) {
                double rad=sqrt((pow(m-SIZE_OF_IMAGE,2) + pow(l-SIZE_OF_IMAGE,2)));
                if (rad < rad1) {
                    if (rad > rad2) {
                        std2 += (pow(value - re(l,m), 2));
                    }
                }
            }
        }
        std2 /= num;
        std2 = sqrt(std2);
        double sca = re(SIZE_OF_IMAGE, SIZE_OF_IMAGE);
        peakToSideLobeRatio = (sca - value) / std2;
    }
};

cv::Ptr<MACE> createMACE(int s) {
    return makePtr<MACEImpl>(s);
}
