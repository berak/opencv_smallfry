#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

/*
GOOD TO KNOW !!!
When accessing a matrix element
    - during creation you need to give rows, cols, type
    - during accessing with the at operator you need to give <type>(rows, cols)
    - during size() collecting you get returned width height, which is cols, rows
    - same goes for creating points, which is cols, rows
*/

/// Calculating the features for a single entry
Mat calculate_features(Mat& input){
    Mat features;

    // Split the image
    vector<Mat> channels;
    split(input, channels);

    // Create a grayscale image
    Mat grayscale;
    cvtColor(input, grayscale, COLOR_BGR2GRAY);
    grayscale.convertTo(grayscale, CV_32F);

    // Calculate the histograms
    int number_bins = 64;
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    bool uniform = true, accum = false;
    Mat b_hist(1,64, CV_32FC1), g_hist(1,64, CV_32FC1), r_hist(1,64, CV_32FC1), gray_hist(1,64, CV_32FC1); //floating point precision needed further down the road
    // OpenCV histogram produces column vectors, but we would like row vectors, therefore we apply a reshape
    calcHist( &channels[0], 1, 0, Mat(), b_hist, 1, &number_bins, &histRange, uniform, accum ); b_hist = b_hist.reshape(1,1);
    calcHist( &channels[1], 1, 0, Mat(), g_hist, 1, &number_bins, &histRange, uniform, accum ); g_hist = g_hist.reshape(1,1);
    calcHist( &channels[2], 1, 0, Mat(), r_hist, 1, &number_bins, &histRange, uniform, accum ); r_hist = r_hist.reshape(1,1);
    calcHist( &grayscale, 1, 0, Mat(), gray_hist, 1, &number_bins, &histRange, uniform, accum ); gray_hist = gray_hist.reshape(1,1);

    // Create a vertical and horizontal spatial histogram
    Mat reduced_vertical, reduced_horizontal;
    // Reduce vertical, from left to right
    reduce(grayscale, reduced_vertical, 1, REDUCE_SUM);
    resize(reduced_vertical, reduced_vertical, Size(1,64));
    transpose(reduced_vertical, reduced_vertical);
    // Reduce horizontal, from top to bottom
    reduce(grayscale, reduced_horizontal, 0, REDUCE_SUM);
    resize(reduced_horizontal, reduced_horizontal, Size(64,1));

    // Normalize the feature vectors
    normalize(b_hist, b_hist, 0, 1, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, 1, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, 1, NORM_MINMAX, -1, Mat() );
    normalize(gray_hist, gray_hist, 0, 1, NORM_MINMAX, -1, Mat() );
    normalize(reduced_vertical, reduced_vertical, 0, 1, NORM_MINMAX, -1, Mat() );
    normalize(reduced_horizontal, reduced_horizontal, 0, 1, NORM_MINMAX, -1, Mat() );

    // Concatenate the feature vectors into single feature vector
    Mat temp, temp2, temp3, temp4;
    hconcat(b_hist, g_hist, temp);
    hconcat(temp, r_hist, temp2);
    hconcat(temp2, gray_hist, temp3);
    hconcat(temp3, reduced_vertical, temp4);
    hconcat(temp4, reduced_horizontal, features);

    return features;
}

int main(int argc, const char** argv)
{
    /// Adding a little help option and command line parser input
    CommandLineParser parser(argc, argv,
        "{ help h usage ?   |       | show this message }"
        "{ positives p      |       | (required) absolute path to positives data folder }"
        "{ negatives n      |       | (required) absolute path to negatives data folder }"
        "{ modelstorage m   |       | (required) absolute path to location for storing the trained models }"
        //"{ testimages t     |       | (required) absolute path to test images folder }"
    );

    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }

    /// Collect data from arguments
    string positives(parser.get<string>("positives"));
    string negatives(parser.get<string>("negatives"));
    string modelstorage(parser.get<string>("modelstorage"));
   // string testimages(parser.get<string>("testimages"));
    if (positives.empty() || negatives.empty() || modelstorage.empty()/* || testimages.empty()*/){
        parser.printMessage();
        return -1;
    }

    /// Collect filenames and store files in a vector
    vector<String> positives_filenames, negatives_filenames;
    glob(positives, positives_filenames);
    glob(negatives, negatives_filenames);

    vector<Mat> positives_;
    vector<Mat> negatives_;
    for(int i = 0; i < (int)positives_filenames.size(); i++){
        Mat temp = imread(positives_filenames[i]);
        positives_.push_back(temp.clone());
    }
    for(int i = 0; i < (int)negatives_filenames.size(); i++){
        Mat temp = imread(negatives_filenames[i]);
        negatives_.push_back(temp.clone());
    }
    cerr << "Data collected from folder structure!" << endl;

    /// Calculate image descriptors of both positives and negatives
    /// Lets start by simply looking at the color histograms from the channels AND the grayscale image
    /// We also added some sort of spatial histogram that has been resized to a fixed size
    // First for the positives
    Mat positive_descriptors, negative_descriptors, combined_descriptors;
    for(int i = 0; i < (int)positives_.size(); i++){
        Mat features = calculate_features(positives_[i]);
        if( i == 0){
            positive_descriptors = features.clone();
        }else{
            vconcat(positive_descriptors.clone(), features, positive_descriptors);
        }
    }
    Mat positive_labels = Mat::ones(positive_descriptors.rows, 1, CV_32SC1);

    // Then for the negatives
    for(int i = 0; i < (int)negatives_.size(); i++){
        Mat features = calculate_features(negatives_[i]);
        if( i == 0){
            negative_descriptors = features.clone();
        }else{
            vconcat(negative_descriptors.clone(), features, negative_descriptors);
        }
    }
    Mat negative_labels = Mat::zeros(negative_descriptors.rows, 1, CV_32SC1);

    /// Combine the data together
    Mat all_data, all_labels;
    vconcat(positive_descriptors, negative_descriptors, all_data);
    vconcat(positive_labels, negative_labels, all_labels);
    cerr << "Features calculated for collected data!" << endl;

    /// Start model training here
    /// Lets first build and SVM model
    cerr << "Starting SVM training ... ";
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(all_data, ROW_SAMPLE, all_labels);
    cerr << "training done!" << endl;

    return 0;
}
