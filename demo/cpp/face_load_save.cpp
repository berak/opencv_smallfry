#include <iostream>
#include "opencv2/core/ocl.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"

using namespace cv;
using namespace cv::face;
using namespace std;

template <class T>
void load_save(const vector<Mat> &images, const vector<int> &labels) {
	Ptr<FaceRecognizer> model1 = T::create();
	String name = model1->getDefaultName();
	model1->train(images,labels);
	model1->write("fr.xml");
	cerr << name << " trained model predicted(2): " << model1->predict(images[2]) << endl;
	Ptr<FaceRecognizer> model2 = T::create();
	model2->read("fr.xml");
	cerr << name << " loaded model is empty: " << model2->empty() << endl;
	cerr << name << " loaded model predicted(2): " << model2->predict(images[2]) << endl << endl;
}
int main() {
	vector<Mat> images;
	vector<int> labels;
	for (int i=0; i<5; i++) {
		Mat m(100,100,CV_8U);
		randu(m,0,255);
		images.push_back(m);
		labels.push_back(i);
	}
	load_save<LBPHFaceRecognizer>(images, labels);
	load_save<FisherFaceRecognizer>(images, labels);
	load_save<EigenFaceRecognizer>(images, labels);
    return 0;
}
