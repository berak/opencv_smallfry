#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace dnn;

// spacing between samples in seconds
const double SAMPLE_DIST = 0.2;

// the input blob structure for the resnet-34_kinetics action recognition,
// taken from:
// https://github.com/opencv/opencv/blob/master/modules/dnn/test/test_onnx_importer.cpp#L611
//
Mat blob5D(const std::vector<Mat> &images) {
    Mat blob0 = blobFromImages(images, 1.0, Size(112, 112), Scalar(114.7748, 107.7354, 99.4750), true, true);

    Net permute;
    LayerParams lp;
    int order[] = {1, 0, 2, 3};
    lp.set("order", DictValue::arrayInt<int*>(&order[0], 4));
    permute.addLayerToPrev("perm", "Permute", lp);
    permute.setInput(blob0);
    Mat input0 = permute.forward().clone();

    int dims[] = {1, 3, 16, 112, 112};
    return input0.reshape(0, 5, &dims[0]);
}

int main(int argc, char** argv) {
    const std::string modelName = "c:/data/dnn/resnet-34_kinetics.onnx";
    const std::string className = "c:/data/dnn/action_recongnition_kinetics.txt";
    std::vector<std::string> classes;
    std::ifstream ifs(className.c_str());
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            classes.push_back(line);
        }
    }

    Net net = readNet(modelName);

    VideoCapture cap(0);

    // Process frames.
    std::vector<Mat> frames;
    Mat frame;
    int64 t0 = getTickCount();
    int64 dt = SAMPLE_DIST * getTickFrequency();
    while (waitKey(10) < 0) {
        cap >> frame;
        if (frame.empty()) {
            return 0;
        }
        int64 t1 = getTickCount();
        if ((t1 - t0 > dt) && (frames.size() < 16)) {
            frames.push_back(frame);
            std::cout << frames.size() << "\r";
            t0 = t1;
        }
        if (frames.size() == 16) {
            //! [Create a 5D(!) blob from a batch of 16 frames]
            Mat blob = blob5D(frames);
            net.setInput(blob);

            Mat prob = net.forward();
            //! [Get the class with the highest score]
            Point classIdPoint;
            double confidence;
            minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
            int classId = classIdPoint.x;
            std::string cls = "";
            if (classes.size()) {
                cls = classes[classId];
            }
            std::cout << format("%4d %3.3f %s", classId, confidence, cls.c_str()) << std::endl;
            frames.clear();
        }
        imshow("AR", frame);
    }
    return 0;
}
