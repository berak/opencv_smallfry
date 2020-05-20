#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/bindings_utils.hpp>
#include <opencv2/quality.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    Mat input = imread("retina.jpg");
    cout << input.size() << " " << input.channels() << " " << input.depth() << " " << typeToString(input.type()) << endl;
    cout << utils::dumpInputArray(input) << endl;
    string basedir = "C:/p/opencv_contrib/modules/quality/samples/";
    Scalar res = quality::QualityBRISQUE::compute(input, basedir + "brisque_model_live.yml", basedir + "brisque_range_live.yml");
    cout << res << endl;
    return 0;
}