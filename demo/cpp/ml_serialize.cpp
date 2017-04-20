#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

template <class T>
void test(const Mat &data, const Mat &labels, const String &fn) {
    cerr << endl;

    Ptr<T> mdl = T::create();
    mdl->train(data,0,labels);
    mdl->save(fn);

    Ptr<T> mdl1 = T::create();
    FileStorage fs(fn,0);
    try {
        mdl1->read(fs.root());
    } catch(cv::Exception &e) {}
    cerr << fn << " " << mdl1->isTrained() << endl;

    Ptr<T> mdl2 = T::create();
    mdl2->read(fs.getFirstTopLevelNode());
    cerr << fn << " " << mdl2->isTrained() << endl;

    Ptr<T> mdl3 = Algorithm::load<T>(fn);
    cerr << fn << " " << mdl3->isTrained() << endl;
}

int main(int argc, char **argv)
{
    Mat data(5,5,CV_32F);
    randu(data,0,1);

    Mat labels_i(5,1,CV_32S,Scalar(0));
    labels_i(Rect(0,2,1,2)) = 1;
    Mat labels_f; labels_i.convertTo(labels_f, CV_32F);
    /*Mat labels_o(5,2,CV_32F, 0.0f); // one-hot for ann
    for (int i=0; i<5; i++) {
        labels_o.at<float>(labels_i.at<int>(i)) = 1;
    }*/

    test<ml::RTrees>(data,labels_i,"rtrees.xml");
    //test<ml::DTrees>(data,labels,"dtrees.xml"); // needs special initialization
    test<ml::SVM>(data,labels_i,"svm.xml");
    test<ml::KNearest>(data,labels_i,"knn.xml");
    test<ml::EM>(data,labels_f,"em.xml");
    test<ml::LogisticRegression>(data,labels_f,"log.xml");
    test<ml::NormalBayesClassifier>(data,labels_i,"bayes.xml");
    //test<ml::ANN_MLP>(data,labels_o,"ann.xml"); // needs layers to work

    FileStorage fs("rtrees.xml", 0);
    FileNode top = fs.root();
    cerr << "root " << top.empty() << " " << top.name() << endl;
    FileNode nx = fs.getFirstTopLevelNode();
    cerr << "top  " << nx.empty() << " " << nx.name() << endl;

}
