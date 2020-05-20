#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

int main() {
    Ptr<ml::TrainData> data = ml::TrainData::loadFromCSV("train.csv", // our file
                                                         1,           // yes, it has a header line
                                                         -1,          // there are labels
                                                         -1,          // and it's the last column
                                                         "",          // we *only* have categorical values (not a mix)
                                                         '\t'         // delimiter
                                                         );
    cout << data->getTrainSamples() << endl;
    cout << data->getTrainResponses().t() << endl;
	return 0;
}