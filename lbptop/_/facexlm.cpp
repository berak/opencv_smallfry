#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#include "landmarks.h"

#include "faceX/face_x.h"
//
// https://github.com/delphifirst/FaceX/
//

struct LandMarks : Landmarks
{
    FaceX face_x;
    LandMarks() : face_x("faceX/model.xml.gz") {}

    virtual int extract(const cv::Mat &img, std::vector<cv::Point> &pt) const
    {
        // originally: 51 landmarks.
        //static int lut[20] = {
        //    0,2,4, 5,7,9,  // eyebrows
        //    19,22, 25,28,  // eyecorners
        //    11,13,14,      // nose
        //    16,18,31,37,42,38,49 // mouth
        //};
        std::vector<cv::Point2d> landmarks = face_x.Alignment(img, cv::Rect(0,0,img.cols,img.rows));
        pt.clear();
        for (size_t i=0; i<landmarks.size(); ++i)
        {
            //pt.push_back(cv::Point(landmarks[lut[i]]));
            pt.push_back(landmarks[i]);
        }
        return int(pt.size());
    }
};

Ptr<Landmarks> createLandmarks() { return makePtr<LandMarks>(); }
