
#include <opencv2/core/core_c.h> // shame, but needed for using dlib
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include "landmarks.h"

// buildocv lbptop dliblm.cpp c:/p/dlib-19.0/dlib/all/source.cpp -Ic:\p\dlib-19.0  -lws2_32 -lwinmm -lcomctl32 -limm32

struct LandMarks : Landmarks
{
    dlib::shape_predictor sp;

    int offset;
    LandMarks(int off=0)
        : offset(off)
    {   // it's only 95mb...
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
    }

    int extract(const cv::Mat &img, std::vector<cv::Point2f> &kp) const
    {
        dlib::rectangle rec(0,0,img.cols,img.rows);
        dlib::full_object_detection shape = sp(dlib::cv_image<uchar>(img), rec);

       // int idx[] = {17,26, 19,24, 21,22, 36,45, 39,42, 38,43, 31,35, 51,33, 48,54, 57,27, 0};

       // for(int k=0; (k<40) && (idx[k]>0); k++)
       //     kp.push_back(cv::Point(crop(shape.part(idx[k]).x(),img.cols),
       //                            crop(shape.part(idx[k]).y(),img.rows)));
       for(int k=0; k<68; k++)
            kp.push_back(cv::Point2f(shape.part(k).x(), shape.part(k).y()));

        return (int)kp.size();
    }
};

cv::Ptr<Landmarks> createLandmarks() { return cv::makePtr<LandMarks>();}



