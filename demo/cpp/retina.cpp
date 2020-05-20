#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/bioinspired.hpp>
using namespace cv;
using namespace std;

// https://answers.opencv.org/upfiles/1572942037928534.jpg

using namespace cv;
using namespace std;
int main(int argc, char** argv) {
    //Mat input = imread("retina.jpg", 0);
    //cout << input.size() << " " << input.channels() << " " << input.depth() << endl;
    //input.convertTo(input, CvType.CV_64F, 1.0/255);

    Ptr<bioinspired::RetinaFastToneMapping> retina = bioinspired::RetinaFastToneMapping::create(Size(320,240));
    //retina->setup();

    // Mat test(input.size(),input.type(), Scalar(90,120,150));
    //retina->applyFastToneMapping(input, test);
    //cout << mean(test) <<  endl;

    //imshow("T",test);
    //waitKey();
    return 0;
}