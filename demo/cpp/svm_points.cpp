#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class exampleClassificationApp {
  public:
    void setup(int w, int h);
    void draw(int w, int h);

    std::vector<cv::Point2f> mPoints;
    std::vector<int> mLabels;
    cv::Ptr<cv::ml::SVM> mSVM;
};

void exampleClassificationApp::setup(int w,int h) {
    for (int i = 0; i < 100; i++) {
        float x = cv::theRNG().uniform(0.0f, float(w));
        float y = cv::theRNG().uniform(0.0f, float(x));
        mPoints.push_back(cv::Point2f(x, y));
        mLabels.push_back(-1.0);
    }
    for (int i = 0; i < 100; i++) {
        float x = cv::theRNG().uniform(0.0f, float(w));
        float y = cv::theRNG().uniform(x, float(h));
        mPoints.push_back(cv::Point2f(x, y));
        mLabels.push_back(1.0);
    }
    cv::Mat trainingData(mPoints.size(), 2, CV_32FC1, mPoints.data());
    cv::Mat trainingLabels(mLabels.size(), 1, CV_32SC1, mLabels.data());

    std::printf("Created SVM\n");
    mSVM = cv::ml::SVM::create();

    std::printf("Setting parameters...\n");
    mSVM->setType(cv::ml::SVM::C_SVC);
    mSVM->setKernel(cv::ml::SVM::LINEAR);
    mSVM->setC(1.0);

    std::printf("Training...\n");
    mSVM->train(trainingData, cv::ml::ROW_SAMPLE, trainingLabels);

    float data[2] = { 0, float(w) / 2.0f };
    cv::Mat query(1, 2, CV_32F, data);
    int response = mSVM->predict(query);

    std::printf("Asked SVM where to classify %f, %f -- result was %d\n", query.at<float>(0,0), query.at<float>(0, 1), response);
}

void exampleClassificationApp::draw(int w, int h) {
    Mat img(h,w,CV_8UC3,Scalar::all(60));
    for (int i = 0; i < mPoints.size(); i++) {
    	Point p = mPoints[i];
    	p.y = h - p.y;
        if (mLabels[i] == -1) {
            circle(img, p, 5, Scalar(200, 0, 0), -1);
        }
        else {
            circle(img, p, 5, Scalar(0,0,200), -1);
        }
    }
    imshow("img", img);
    waitKey();
    cv::Mat supports;
    if (mSVM->isTrained()) {
        supports = mSVM->getSupportVectors();
        std::cout << "Support Vectors are " << supports << std::endl;
        supports = mSVM->getUncompressedSupportVectors();
        std::cout << "uncompressed Support Vectors are " << supports << std::endl;
        mSVM->save("svm_points.xml");
    }

}

int main() {
	exampleClassificationApp app;
	app.setup(400,400);
	app.draw(400,400);
	return 0;
}
