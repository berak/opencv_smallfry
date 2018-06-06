#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    vector<String> fn;
    glob("c:/data/cat-dog/*.jpg", fn, true);
    // 700 cats, 699 dogs
    //glob() will also sort names lexically, so the cats come first!

    std::string modelTxt = "c:/data/mdl/squeezenet/deploy.prototxt";
    std::string modelBin = "c:/data/mdl/squeezenet/squeezenet_v1.1.caffemodel";
    dnn::Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
    cv::Size inputImgSize = cv::Size(227, 227);

    Mat_<int> layers(4, 1);
    layers << 1000, 400, 100, 2; // the sqeezenet pool10 layer has 1000 neurons

    Ptr<ml::ANN_MLP> nn = ml::ANN_MLP::create();
    nn->setLayerSizes(layers);
    nn->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);
    nn->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
    nn->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 300, 0.0001));

    Mat train, test;
    Mat labels(1199,2,CV_32F,0.f); // 1399 - 200 test images
    for (size_t i=0; i<fn.size(); i++) {
        // use the dnn as a "fixed function" preprocessor (no training here)
        Mat img = imread(fn[i]);
        net.setInput(dnn::blobFromImage(img, 1, inputImgSize, Scalar::all(127), false));
        Mat blob = net.forward("pool10");
        Mat f = blob.reshape(1,1).clone();

        if (i<100) {
            // test cat
            test.push_back(f);
        } else
        if (i>=100 && i<700) {
            // train cat
            train.push_back(f);
            labels.at<float>(i-100,0) = 1;
        } else
        if (i>=700 && i<800) {
            // test dog
            test.push_back(f);
        } else {
            // train dog
            train.push_back(f);
            labels.at<float>(i-200,1) = 1;
        }
        cout << i << "\r";
    }

    cout << train.size() << " " << labels.size() << " " << test.size() << endl;
    nn->train(train, 0, labels);
    nn->save("cats.dogs.ann.yml.gz");

    Mat result;
    nn->predict(test,result);
    float correct_cat = 0;
    float correct_dog = 0;
    for (int i=0; i<100; i++)
        correct_cat += result.at<float>(i,0) > result.at<float>(i,1); // true cat
    for (int i=100; i<200; i++)
        correct_dog += result.at<float>(i,1) > result.at<float>(i,0); // true dog;
    float acc = (correct_cat + correct_dog) / 200;
    cout << correct_cat << " " << correct_dog << " : " << acc << endl;
cout << result;
    return 0;
}
