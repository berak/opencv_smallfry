#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    String face_cascade_name = argc>1 ? argv[1] : "c:/p/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
    Mat im = imread("c:/data/img/people.jpg");
    resize(im,im,Size(), 4,4);

    cv::imshow("src", im);
    cv::waitKey(1);

    CascadeClassifier face_cascade(face_cascade_name);
    cerr << face_cascade.empty() << endl;

    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( im, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    cerr << faces.size() << " objects detected." << endl;

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        rectangle( im, faces[i], Scalar( 255, 0, 255 ), 4, 8, 0 );
    }

    cv::imshow("dst", im);
    cv::waitKey(1);

    //-- Detect faces, now with weights
    vector<int> rejectLevels;
    vector<double> rejectWeights;
    face_cascade.detectMultiScale( frame_gray, faces, rejectLevels, rejectWeights,  1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30), Size(90, 90), true );
    cerr << faces.size() << " objects detected." << endl;

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        rectangle( im, faces[i], Scalar( 0, 255, 0 ), 2, 8, 0 );
        cerr << rejectLevels[i] << " " << rejectWeights[i] << "\t" << faces[i] << endl;
    }
    cerr << faces.size() << " objects detected." <<  endl;

    cv::imshow("dst_weights", im);
    cv::waitKey();
    return 0;
}
