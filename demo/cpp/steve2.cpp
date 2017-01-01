#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

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

    return features; // flat row.
}

int main(int argc, const char** argv)
{
    /// Adding a little help option and command line parser input
    CommandLineParser parser(argc, argv,
        "{ help h usage ?   |       | show this message }"
        "{ testimages t     |       | (required) absolute path to test images folder }"
        "{ testlabel l      |-1     | (0 or 1) images folder contains this class only\n"
        "                -1 : assume 'pos' in filename as a label }"
    );

    if (parser.has("help")){
        parser.printMessage();
        cerr << "For now the models are hard coded read in, not to have to pass a tons of extra arguments." << endl;
        return 0;
    }

    /// Collect data from arguments
    string testimages(parser.get<string>("testimages"));
    int testlabel(parser.get<int>("testlabel"));
     if (testimages.empty()){
        parser.printMessage();
        return -1;
    }

    /// Read in the image samples
    vector<String> filenames;
    glob(testimages, filenames);

    /// Read in the trained models
    Ptr<SVM> svm_model = StatModel::load<SVM>("model_SVM.xml");
    cout << "Svm isTrained: " << svm_model->isTrained() << endl;

    ///berak:
    // we need ground truth labels here, to measure the prediction acc.
    Mat truth;
    /// Read images sequentially, calculate their features and provide a label output
    cerr << "Starting feature calculation ... ";
    Mat feature_descriptors;
    for(int i = 0; i < (int)filenames.size(); i++){
        Mat image = imread(filenames[i]);
        if (image.empty())
            continue;

        Mat features = calculate_features(image);
        feature_descriptors.push_back(features);

        // find ground truth
        if (testlabel != -1) {
            truth.push_back(float(testlabel)); // all the same
        } else {
            int l = filenames[i].find("pos") > -1; // from filename
            truth.push_back(float(l));
        }
    }
    cerr << "precalculation done!" << endl;

    cerr << "Classifying samples using SVM model ... " << endl;
    Mat pred;
    svm_model->predict(feature_descriptors, pred);

    Mat_<int> confusion(2,2, 0);
    for (size_t i=0; i<pred.total(); i++)
    {
        int r = int(pred.at<float>(i));
        int c = int(truth.at<float>(i));
        confusion(r, c) ++;
    }
    cout << confusion << endl;

    int nz = sum(confusion.diag())[0];
    int nelm = sum(confusion)[0];
    float acc =  (float(nz) / nelm);
    cout << nz << "/" << nelm << " : " << acc << endl;
    return 0;
}
