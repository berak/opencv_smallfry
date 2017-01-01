#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/bioinspired.hpp"

using namespace std;
using namespace cv;

int main()
{
    Mat img = imread("lighting.jpg", 0); // grayscale
    imshow("orig", img);

    Mat imh;
    equalizeHist(img, imh);
    imshow("equalize", imh);

    Mat imc;
    Ptr<CLAHE> cl = createCLAHE(30);
    cl->apply(img, imc);
    imshow("clahe", imc);

    Ptr<bioinspired::Retina> retina(bioinspired::createRetina(img.size()));
    //// (realistic setup)
    bioinspired::RetinaParameters ret_params;
    ret_params.OPLandIplParvo.horizontalCellsGain = 1.7f;
    ret_params.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity = 1.79f;
    ret_params.OPLandIplParvo.ganglionCellsSensitivity = 1.7f;
    retina->setup(ret_params);

    Mat imr;
    retina->run(img);
    retina->getParvo(imr);
    imshow("retina", imr);

    waitKey();

    return 0;
}
