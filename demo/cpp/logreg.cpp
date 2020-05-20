#include <opencv2/ml.hpp>

using namespace cv;
using namespace ml;
using namespace std;

Ptr<LogisticRegression> model(float learningRate, int iterations, int miniBatchSize) {
    Ptr<LogisticRegression> logreg = LogisticRegression::create();

    logreg->setLearningRate(learningRate);
    logreg->setIterations(iterations);
    logreg->setMiniBatchSize(miniBatchSize);
    logreg->setTrainMethod(LogisticRegression::BATCH);
    logreg->setRegularization(LogisticRegression::REG_L2);

    return logreg;
}

int main(int, char**)
{

    const Ptr<TrainData> trainData = TrainData::loadFromCSV("train_cleaned.csv",
        1, // lines to skip
        0, // index of label
        -1 // 1 response per line
    );

    trainData->setTrainTestSplitRatio(0.8);

    Ptr<LogisticRegression> logreg = model(0.8, 10001, 10);
    // Ptr<StatModel> logreg = SVM::create(); //model(0.001, 10, 1);



    logreg->train(trainData);

    Mat results;

    logreg->predict(trainData->getTestSamples(), results);

    cout << results.t() << endl;

    return 0;
}