#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

using namespace std;
using namespace cv;

void createQuaternionImage(InputArray _img, OutputArray _qimg)
{
    int type = _img.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert((depth == CV_8U || depth == CV_32F || depth == CV_64F) && _img.dims() == 2 && cn == 3);
    vector<Mat> qplane(4);
    vector<Mat> plane;
    split(_img, plane);
    qplane[0] = Mat::zeros(_img.size(), CV_64FC1);
    for (int i = 0; i < cn; i++)
        plane[i].convertTo(qplane[i + 1], CV_64F);
    merge(qplane, _qimg);
}

void qconj(InputArray _img, OutputArray _qimg)
{
    int type = _img.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert((depth == CV_32F || depth == CV_64F) && _img.dims() == 2 && cn == 4);
    vector<Mat> qplane(4),plane;
    split(_img, plane);
    qplane[0] = plane[0].clone();
    qplane[1] = -plane[1].clone();
    qplane[2] = -plane[2].clone();
    qplane[3] = -plane[3].clone();
    merge(qplane, _qimg);
}

void qunitary(InputArray _img, OutputArray _qimg)
{
    int type = _img.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert((depth == CV_32F || depth == CV_64F) && _img.dims() == 2 && cn == 4);
    vector<Mat> qplane(4), plane;
    split(_img, plane);
    qplane[0] = plane[0].clone();
    qplane[1] = -plane[1].clone();
    qplane[2] = -plane[2].clone();
    qplane[3] = -plane[3].clone();
    float *ptr0 = qplane[0].ptr<float>(0, 0), *ptr1 = qplane[1].ptr<float>(0, 0);
    float *ptr2 = qplane[2].ptr<float>(0, 0), *ptr3 = qplane[3].ptr<float>(0, 0);
    int nb = plane[0].rows*plane[0].cols;
    for (int i = 0; i < nb; i++, ptr0++, ptr1++, ptr2++, ptr3++)
    {
        float d = *ptr0 * *ptr0 + *ptr1 * *ptr1 + *ptr2 * *ptr2 + *ptr3 * *ptr3;
        d = 1; // sqrt(d);
        *ptr0 *= d;
        *ptr1 *= d;
        *ptr2 *= d;
        *ptr3 *= d;

    }
    merge(qplane, _qimg);
}

void QDFT(InputArray _img, OutputArray _qimg, int  	flags,bool sideLeft )
{
//    CV_INSTRUMENT_REGION()

    int type = _img.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(depth == CV_64F  && _img.dims() == 2 && cn == 4);
    float c;
    if (sideLeft)
        c = 1;  // Left qdft
    else
        c = -1; // right qdft

    vector<Mat> q;
    Mat img;
    img = _img.getMat();

    CV_Assert(getOptimalDFTSize(img.rows) == img.rows && getOptimalDFTSize(img.cols) == img.cols);

    split(img, q);
    Mat c1r;
    Mat c1i; // Imaginary part of c1 =x'
    Mat c2r; // Real part of c2 =y'
    Mat c2i; // Imaginary part of c2=z'
    c1r = q[0].clone();
    c1i = (q[1] + q[2] + q[3]) / sqrt(3);
    c2r = (q[2] - q[3]) / sqrt(2);
    c2i = c*(q[3] + q[2] - 2 * q[1]) / sqrt(6);
    vector<Mat> vc1 = { c1r,c1i }, vc2 = { c2r,c2i };
    Mat c1, c2,C1,C2;
    merge(vc1, c1);
    merge(vc2, c2);
    if (flags& DFT_INVERSE)
    {
        dft(c1, C1, DFT_COMPLEX_OUTPUT| DFT_INVERSE );
        dft(c2, C2, DFT_COMPLEX_OUTPUT| DFT_INVERSE );
    }
    else
    {
        dft(c1, C1, DFT_COMPLEX_OUTPUT );
        dft(c2, C2, DFT_COMPLEX_OUTPUT );
    }
    split(C1, vc1);
    split(C2, vc2);
    vector<Mat> qdft(4);
    qdft[0] = vc1[0].clone();
    qdft[1] = vc1[1] / sqrt(3) - 2*vc2[1]/sqrt(6);
    qdft[2] = vc1[1] / sqrt(3) + vc2[0] / sqrt(2) + vc2[1] / sqrt(6);
    qdft[3] = c*(vc1[1] / sqrt(3) - vc2[0] / sqrt(2) + vc2[1] / sqrt(6));
    Mat dst0;
    merge(qdft, dst0);
    dst0.copyTo(_qimg);
}

void qmultiply(InputArray  	src1, InputArray  	src2, OutputArray  	dst)
{
    int type = src1.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(depth == CV_64F && src1.dims() == 2 && cn == 4);
    type = src2.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(depth == CV_64F && src2.dims() == 2 && cn == 4);
    vector<Mat> q3(4);
    if (src1.rows() == src2.rows() && src1.cols() == src2.cols())
    {
        vector<Mat> q1, q2;
        split(src1, q1);
        split(src2, q2);
        q3[0] = q1[0].mul(q2[0]) - q1[1].mul(q2[1]) - q1[2].mul(q2[2]) - q1[3].mul(q2[3]);
        q3[1] = q1[0].mul(q2[1]) + q1[1].mul(q2[0]) + q1[2].mul(q2[3]) - q1[3].mul(q2[2]);
        q3[2] = q1[0].mul(q2[2]) - q1[1].mul(q2[3]) + q1[2].mul(q2[0]) + q1[3].mul(q2[1]);
        q3[3] = q1[0].mul(q2[3]) + q1[1].mul(q2[2]) - q1[2].mul(q2[1]) + q1[3].mul(q2[0]);
    }
    else if (src1.rows() == 1 && src1.cols() == 1)
    {
        vector<Mat> q2;
        Vec4d q1 = src1.getMat().at<Vec4d>(0, 0);
        split(src2, q2);
        q3[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
        q3[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
        q3[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
        q3[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
    }
    else if (src2.rows() == 1 && src2.cols() == 1)
    {
        vector<Mat> q1;
        split(src1, q1);
        Vec4d q2 = src2.getMat().at<Vec4d>(0, 0);
        q3[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
        q3[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
        q3[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
        q3[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
    }
    else
        CV_Assert(src1.rows() == src2.rows() && src1.cols() == src2.cols());
    merge(q3, dst);

}

void colorMatchTemplate(InputArray _image, InputArray _templ, OutputArray _result)
{
    Mat image = _image.getMat(),imageF;
    Mat colorTemplate = _templ.getMat();
    int rr = getOptimalDFTSize(image.rows);
    int cc = getOptimalDFTSize(image.cols);
    rr = max(rr, cc);
    cc = getOptimalDFTSize(colorTemplate.rows);
    rr = max(rr, cc);
    cc = getOptimalDFTSize(colorTemplate.cols);
    rr = max(rr, cc);
    Mat logo(rr, rr, CV_64FC3, Scalar::all(0));
    Mat img = Mat(rr, rr, CV_64FC3, Scalar::all(0));
    Scalar x = mean(colorTemplate);
    colorTemplate.convertTo(colorTemplate, CV_64F, 1 / 256.),
    subtract(colorTemplate, x / 256., colorTemplate);
    colorTemplate.copyTo(logo(Rect(0, 0, colorTemplate.cols, colorTemplate.rows)));
    image.convertTo(imageF, CV_64F, 1 / 256.);
    subtract(imageF, x / 256., imageF);
    imageF.copyTo(img(Rect(0, 0, image.cols, image.rows)));
    Mat qimg, qlogo;
    Mat qimgFFT, qimgIFFT, qlogoFFT;
    // Create quaternion image
    createQuaternionImage(img, qimg);
    createQuaternionImage(logo, qlogo);
    // quaternion fourier transform
    QDFT(qimg, qimgFFT, 0, true);
    QDFT(qimg, qimgIFFT, DFT_INVERSE, true);
    QDFT(qlogo, qlogoFFT, 0, false);
    double sqrtnn = sqrt(static_cast<int>(qimgFFT.rows*qimgFFT.cols));
    qimgFFT /= sqrtnn;
    qimgIFFT *= sqrtnn;
    qlogoFFT /= sqrtnn;
    Mat mu(1, 1, CV_64FC4, Scalar(0, 1, 1, 1));
    Mat qtmp, qlogopara, qlogoortho;
    qmultiply(mu, qlogoFFT, qtmp);
    qmultiply(qtmp, mu, qtmp);
    subtract(qlogoFFT, qtmp, qlogopara);
    qlogopara = qlogopara / 2;
    subtract(qlogoFFT, qlogopara, qlogoortho);
    Mat qcross1, qcross2, cqf, cqfi;
    qconj(qimgFFT, cqf);
    qconj(qimgIFFT, cqfi);
    qmultiply(cqf, qlogopara, qcross1);
    qmultiply(cqfi, qlogoortho, qcross2);
    Mat pwsp = qcross1 + qcross2;
    Mat crossCorr, pwspUnitary;
    qunitary(pwsp, pwspUnitary);
    QDFT(pwspUnitary, crossCorr, DFT_INVERSE, true);
    vector<Mat> p;
    split(crossCorr, p);
    Mat imgcorr = (p[0].mul(p[0]) + p[1].mul(p[1]) + p[2].mul(p[2]) + p[3].mul(p[3]));
    sqrt(imgcorr, _result);

}


void AddSlider(String sliderName, String windowName, int minSlider, int maxSlider, int valDefault, int *valSlider, void(*f)(int, void *), void *r)
{
    createTrackbar(sliderName, windowName, valSlider, 1, f, r);
    setTrackbarMin(sliderName, windowName, minSlider);
    setTrackbarMax(sliderName, windowName, maxSlider);
    setTrackbarPos(sliderName, windowName, valDefault);
}

struct SliderData {
    Mat img;
    int thresh;
};

void UpdateThreshImage(int x, void *r)
{
    SliderData *p = (SliderData*)r;
    Mat dst;

    threshold(p->img, dst, p->thresh, 255, THRESH_BINARY);
    imshow("Max Quaternion corr",dst);
}

int main(int argc, char *argv[])
{
#define TESTMATCHING

#ifdef TESTMATCHING
    Mat imgLogo = imread("c:/p/opencv/samples/data/opencv-logo.png", IMREAD_COLOR);
    Mat fruits = imread("c:/p/opencv/samples/data/lena.jpg", IMREAD_COLOR);
//    fruits = fruits * 0;
    resize(fruits,fruits, Size(), 0.5, 0.5);
    Mat img,colorTemplate;
    imgLogo(Rect(0, 0, imgLogo.cols, 580)).copyTo(img);
    resize(img, colorTemplate, Size(), 0.05, 0.05);
    vector<Mat> colorMask(4);
    inRange(colorTemplate, Vec3b(255, 255, 255), Vec3b(255, 255, 255), colorMask[0]);
//    colorTemplate.setTo(Scalar(0,0,0), colorMask[0]);

    inRange(colorTemplate, Vec3b(255, 0, 0), Vec3b(255, 0, 0), colorMask[0]);
    inRange(colorTemplate, Vec3b(0, 255, 0), Vec3b(0,255,  0), colorMask[1]);
    inRange(colorTemplate, Vec3b( 0, 0,255), Vec3b( 0, 0,255), colorMask[2]);
    colorMask[3] = Mat(colorTemplate.size(), CV_8UC3, Scalar(255));
    RNG r;
    for (int i = 0; i < 16; i++)
    {
        Point p(i / 4 * 65+10, (i % 4) * 65+10);
        Mat newLogo= colorTemplate.clone();
        if (i % 3 != 2)
        {
            newLogo.setTo(Scalar(r.uniform(0, 256), r.uniform(0, 256), r.uniform(0, 256)), colorMask[i % 4]);
            newLogo.setTo(Scalar(r.uniform(0, 256), r.uniform(0, 256), r.uniform(0, 256)), colorMask[(i + 1) % 4]);
        }
            newLogo.copyTo(fruits(Rect(p.x, p.y, colorTemplate.cols, colorTemplate.rows)));

    }
#else
    Mat fruits = imread("15214713881857319.png", IMREAD_COLOR);
    Mat colorTemplate = imread("15214714019776815.png", IMREAD_COLOR);
    Mat img= colorTemplate;
#endif
    imshow("Image", fruits);
    imshow("opencv_logo", colorTemplate);

    if (img.empty())
    {
        cout << "Cannot load image file\n";
        return 0;
    }
    Mat imgcorr;
    SliderData p;
    colorMatchTemplate(fruits, colorTemplate, imgcorr);
    normalize(imgcorr, imgcorr,1,0,NORM_MINMAX);
    imgcorr.convertTo(p.img, CV_8U, 255);
    imshow("quaternion correlation", imgcorr);
    int level = 200;
    AddSlider("Level", "quaternion correlation", 0, 255, p.thresh, &p.thresh, UpdateThreshImage, &p);
    int code = 0;
    while (code != 27)
    {
        code = waitKey(50);
    }

    FileStorage fs("corr.yml", FileStorage::WRITE);
    fs<<"Image"<< imgcorr;
    fs.release();
    waitKey(0);
    return 0;
}
