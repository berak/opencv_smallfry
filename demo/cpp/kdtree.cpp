#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include "opencv2/core/core_c.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

using cv::Ptr;

int main(){

    Mat xTrainData = (Mat_<float>(5,2) << 1, 1.1, 1.1, 1, 2, 2, 2.1, 2, 2.1, 2.1);
    Mat yTrainLabels = (Mat_<float>(5,1) << 1, 1, 2, 2, 2);

    cout << "===============xTrainData===================="<< endl;
    cout << "xTrainData: "  << xTrainData<< endl;

    cout << "===============yTrainLabels===================="<< endl;
    cout << "yTrainLabels: "  << yTrainLabels << endl;

    // KNearest KDTree implementation
    Ptr<KNearest> knnKdt = KNearest::create();
    knnKdt->setIsClassifier(false);

    // When runs to this step, it does not run any more.
    knnKdt->train(xTrainData, ml::ROW_SAMPLE, yTrainLabels);

    Mat xTestData = (Mat_<float>(2,2) << 1.1, 1.1, 2, 2.2);
    //Mat xTestData(3,2,CV_32F); randu(xTestData,0,3);
    Mat zBestLabels, neighbours, dist;
    cout << "===============xTestData===================="<< endl;
    cout << "xTestData: "  << xTestData << endl;


    int K = 4;
    cout << "-------------BFORCE---------------------------\n";
    knnKdt->findNearest(xTestData, K, zBestLabels, neighbours, dist);
    cout << zBestLabels << endl;
    cout << neighbours << endl;
    cout << dist << endl;
    zBestLabels.release();
    neighbours.release();
    dist.release();

    cout << "-------------KDTREE---------------------------\n";
    knnKdt->setAlgorithmType(KNearest::KDTREE);
    knnKdt->setIsClassifier(true);

   // When runs to this step, it does not run any more.
    knnKdt->train(xTrainData, ml::ROW_SAMPLE, yTrainLabels);
    knnKdt->findNearest(xTestData, K, zBestLabels, neighbours, dist);
    cout << zBestLabels << endl;
    cout << neighbours << endl;
    cout << dist << endl;

}
