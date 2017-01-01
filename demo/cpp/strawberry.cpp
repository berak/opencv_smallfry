#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

/// CONCEPTS THAT SHOULD BE STUDIED IN THIS LAB
/// Creating a key click selector for retrieving pixel information
/// Nearest Neighbor / Normal Bayes Classifier / SVM Classifier

void on_mouse(int event, int x, int y, int , void * param){
    vector<Point>* _collection = (vector<Point>*)param;
    if(event == EVENT_LBUTTONDOWN){
        cerr << "Point selected at [" << x << "," << y << "]." << endl << endl;
        _collection->push_back(Point(x,y));
    }
    if(event == EVENT_RBUTTONDOWN ){
        if(_collection->size() == 0){
            cerr << "Nothing more to remove!" << endl << endl;
        }else{
            cerr << "Last point removed from collection." << endl << endl;
            _collection->pop_back();
        }
    }
    if(event == EVENT_MBUTTONDOWN){
        if(_collection->size() == 0){
            cerr << "Collection is empty, first add points!" << endl << endl;
        }else{
            cerr << "Current contents of collection of points: ";
            for(int i = 0; i < _collection->size(); i++){
                cerr << _collection->at(i) << " ";
            }
            cerr << endl << endl;
        }
    }
}

int main( int argc, const char** argv )
{
    /// Use the cmdlineparser to process input arguments
    CommandLineParser parser(argc, argv,
        "{ help h       |      | show this message }"
        "{ image i      |      | (required) path to image }"
    );

    /// If help is entered
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }

    /// Parse arguments
    string image_location(parser.get<string>("image"));
    if (image_location.empty()){
        parser.printMessage();
        return -1;
    }

    /// Load image and let the user select pixels
    Mat img = imread(image_location, IMREAD_COLOR);
    GaussianBlur(img, img, Size(5,5), 0);

    /// Get strawberry pixels
    cerr << "********************************************************************" << endl;
    cerr << "SELECTING STRAWBERRY LOCATIONS" << endl;
    cerr << "********************************************************************" << endl;
    vector<Point> strawberry;
    Mat canvas1 = img.clone();
    stringstream stream1; stream1 << "Selecting strawberry locations!";
    putText(canvas1, stream1.str(), Point(50, img.rows - 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,0));
    imshow("input_image", canvas1); setMouseCallback("input_image", on_mouse, (void*)(&strawberry)); waitKey(0);

    /// Get background pixels
    cerr << "********************************************************************" << endl;
    cerr << "SELECTING BACKGROUND LOCATIONS" << endl;
    cerr << "********************************************************************" << endl;
    vector<Point> background;
    Mat canvas2 = img.clone();
    stringstream stream2; stream2 << "Selecting background locations!";
    putText(canvas2, stream2.str(), Point(50, img.rows - 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,0));
    imshow("input_image", canvas2); setMouseCallback("input_image", on_mouse, (void*)(&background)); waitKey(0);

    /// Collecting the pixel information
    cerr << "********************************************************************" << endl;
    cerr << "Calculating descriptors for each position" << endl;
    cerr << "********************************************************************" << endl;

    // Retrieving the HSV pixel values for each position
    Mat img_hsv;
    cvtColor(img, img_hsv, COLOR_BGR2HSV);
    vector<Vec3b> descriptors;
    for(int i=0; i < strawberry.size(); i++){
        descriptors.push_back( img_hsv.at<Vec3b>(strawberry[i].y, strawberry[i].x) );
    }
    for(int i=0; i < background.size(); i++){
        descriptors.push_back( img_hsv.at<Vec3b>(background[i].y, background[i].x) );
    }

    // Putting them into the correct data format for SVM training
    Mat labels(descriptors.size(),1, CV_32SC1);
    Mat trainingData(descriptors.size(), 3, CV_32FC1);
    for(int i=0; i < descriptors.size(); i++){
        Vec3b temp = descriptors[i];
        trainingData.at<float>(i,0) = temp[0]; trainingData.at<float>(i,1) = temp[1]; trainingData.at<float>(i,2) = temp[2];
        if( i < strawberry.size() ){
            labels.at<int>(i,0) = 1;
        }else{
            labels.at<int>(i,0) = 0;
        }
    }

    cerr << trainingData << endl;
    cerr << labels << endl;

    /// Training the classifier
    cerr << "********************************************************************" << endl;
    cerr << "Training classifiers" << endl;
    cerr << "********************************************************************" << endl;

    /// ---------------------------
    /// Nearest Neighbor Classifier
    /// ---------------------------
    cerr << "Training a 1 Nearest Neighbor Classifier ... " << endl;
    Ptr<KNearest> kNN = KNearest::create();
    Ptr<TrainData> trainingDataKNN = TrainData::create(trainingData, ROW_SAMPLE, labels);
    kNN->setIsClassifier(true);                             // Set a classification instead of a regression
    kNN->setAlgorithmType(KNearest::BRUTE_FORCE);    // Implementation of KNearest
    kNN->setDefaultK(1);                                    // Number of nearest samples to classify
    kNN->train(trainingDataKNN);

    /// -----------------------
    /// Normal Bayes Classifier
    /// -----------------------
    cerr << "Training a Normal Bayes Classifier ... " << endl;
    Ptr<NormalBayesClassifier> normalBayes = NormalBayesClassifier::create();
    normalBayes->train(trainingData, ROW_SAMPLE, labels);

    /// ----------------------
    /// Support Vector Machine
    /// ----------------------
    cerr << "Training a Support Vector Machine Classifier ... " << endl;
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingData, ROW_SAMPLE, labels);

    Mat templabels;
    normalBayes->predict(trainingData, templabels);
    cerr << "Output of trained normalBayes on training data:" << endl;
    cerr << templabels << endl;

    Mat templabels2;
    svm->predict(trainingData, templabels2);
    cerr << "Output of trained SVM on training data:" << endl;
    cerr << templabels2 << endl;

    waitKey(0);

    /// Running the classifier on the whole image
    cerr << "********************************************************************" << endl;
    cerr << "Running the classifier on the input image and creating masks" << endl;
    cerr << "********************************************************************" << endl;
    Mat data_test(1,3,CV_32FC1);
    Mat labels_kNN, labels_normalBayes, labels_SVM;
    Mat mask_kNN = Mat::zeros(img.rows, img.cols, CV_8UC1), mask_normalBayes = Mat::zeros(img.rows, img.cols, CV_8UC1), mask_SVM = Mat::zeros(img.rows, img.cols, CV_8UC1);
    for(int i=0; i < img.rows; i++){
        for(int j=0; j < img.cols; j++){
            // Get the data descriptor on that location --> for now 3 HSV pixel values
            Vec3b pixelvalue = img_hsv.at<Vec3b>(i,j);
            data_test.at<float>(0,0) = pixelvalue[0];
            data_test.at<float>(0,1) = pixelvalue[1];
            data_test.at<float>(0,2) = pixelvalue[2];

            // Run the classifiers
            normalBayes->predict(data_test, labels_normalBayes);
            svm->predict(data_test, labels_SVM);

            // Fill the corresponding masks
            // ATTENTION
            // NaiveBayes --> output resultaten als integer labels 32S -> int jump
            // SVM --> output resultaten als floating point labels 32F -> float jump
            mask_normalBayes.at<uchar>(i,j) = labels_normalBayes.at<int>(0,0);
            mask_SVM.at<uchar>(i,j) = labels_SVM.at<float>(0,0);
        }
    }

    imshow("nb", mask_normalBayes*255);
    imshow("svm", mask_SVM*255);
    waitKey(0);

    vector<Mat> channels;
    split(img, channels);

    Mat nb_b, nb_g, nb_r;
    Mat svm_b, svm_g, svm_r;
    multiply(channels[0], mask_normalBayes, nb_b); multiply(channels[1], mask_normalBayes, nb_g); multiply(channels[2], mask_normalBayes, nb_r);
    multiply(channels[0], mask_normalBayes, svm_b); multiply(channels[1], mask_normalBayes, svm_g); multiply(channels[2], mask_normalBayes, svm_r);

    //Mat result_kNN(img.rows, img.cols, CV_8UC3);
    //Mat in_kNN[] = { channels[0] * mask_kNN, channels[1] * mask_kNN, channels[2] * mask_kNN };
    int from_to[] = { 0,0, 1,1, 2,2 };
    //mixChannels( in_kNN, 3, &result_kNN, 1, from_to, 3 );

    Mat result_normalBayes(img.rows, img.cols, CV_8UC3);
    Mat in_normalBayes[] = { nb_b, nb_g, nb_r };
    mixChannels( in_normalBayes, 3, &result_normalBayes, 1, from_to, 3 );

    Mat result_SVM(img.rows, img.cols, CV_8UC3);
    Mat in_SVM[] = { svm_b, svm_g, svm_r };
    mixChannels( in_SVM, 3, &result_SVM, 1, from_to, 3 );

    //imshow("result kNN", result_kNN);
    imshow("result Normal Bayes", result_normalBayes);
    imshow("result SVM", result_SVM);
    waitKey(0);

    return 0;
}
