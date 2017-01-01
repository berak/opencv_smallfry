
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;


Mat color_hist(const Mat &in, int grid)
{
    Mat feature;
    Mat hsv, hue;
    cvtColor(in,hsv,COLOR_BGR2HSV);
    extractChannel(hsv, hue, 0);
    int w = cvCeil(float(in.cols)/grid);
    int h = cvCeil(float(in.rows)/grid);
    for (int j=0; j<in.rows-h; j+=h)
    {
        for (int i=0; i<in.cols-w; i+=w)
        {
            Rect r(i,j,w,h);
            Mat roi(hue, r);
            Mat hist;
            int histSize = 32;
            float range[] = { 0, 256 } ;
            const float *histRange = { range };
            calcHist( &roi, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
            feature.push_back(hist.reshape(1,1));
        }
    }
    return feature.reshape(1,1);
}


int main(int argc, char* argv[]) {

    int G = 5;
    vector<String> fn;
    glob("c:/data/img/cache", fn, true);
    Mat features;
    if (0)
    {
        for (size_t i=0; i<fn.size(); i++)
        {
            Mat img = imread(fn[i]);
            if (img.empty()) continue;
            Mat c = color_hist(img, G);
            cout << i << "/" << fn.size() << "\r";
            features.push_back(c);
        }
        FileStorage fs("color_hist.xml",1);
        fs << "hists" << features;
        fs.release();
    }
    else
    {
        FileStorage fs("color_hist.xml",0);
        fs["hists"] >> features;
        fs.release();
    }
    cout << "\n" << features.size() << endl;
//    features.convertTo(features,CV_32F);


    Mat img = imread("im/balls.png");
    Mat query = color_hist(img,G);


    double radius= 2000.0;
    unsigned int max_neighbours = 5;
    cv::Mat indices, dists;
    cv::flann::Index flann_index(features, cv::flann::HierarchicalClusteringIndexParams(), cvflann::FLANN_DIST_L2);
    //cv::flann::Index flann_index(features, cv::flann::LinearIndexParams(), cvflann::FLANN_DIST_L2);

    int K=5;
    cv::flann::SearchParams params;
    flann_index.knnSearch(query, indices, dists, K, params);

//    flann_index.radiusSearch(query, indices, dists, radius, max_neighbours, cv::flann::SearchParams(32));
    for (size_t i=0; i<indices.total(); i++) {
        if (dists.at<float>(i) == 0) break;
        Mat im = imread(fn[indices.at<int>(i)]);
        imshow(format("%d",i),im);
    }
    cout << indices.type() << endl << indices << endl;
    cout << dists.type() << endl << dists << endl;
    waitKey();
    return 0;
}
