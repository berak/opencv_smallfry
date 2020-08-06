/**
 * Program that runs pushbroom stereo.
 * Copyright 2013-2015, Andrew Barry <abarry@csail.mit.edu>
 *
 */

//#include "pushbroom-stereo.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/calib3d.hpp>
#include "pushbroom_stereo.hpp"
#include <iostream>
#include <fstream>


using namespace cv;
using std::cout;
using std::endl;


Mat ident(int h, int w) {
    Mat_<Point2f> m(h,w);
    for (int i=0; i<h; i++)
        for (int j=0; j<w; j++)
            m(i,j)=Point2f(j,i);
    return m;
}

int main() {
    // initilize default parameters
    pushbroom::StereoState state;

    // https://github.com/andybarry/flight/blob/master/sensors/stereo/deltawing-stereo-odroid-cam1.conf
    state.disparity = -33;
    state.zero_dist_disparity = -21;
    state.sobelLimit = 860;
    state.horizontalInvarianceMultiplier = 0.5;
    state.blockSize = 5;
    state.sadThreshold = 54;


    if (state.blockSize > 10 || state.blockSize < 1)
    {
        fprintf(stderr, "Warning: block size is very large "
            "or small (%d).  Expect trouble.\n", state.blockSize);
    }
    if (0) {
        auto read_map = [](int k, Mat &dst) {
            Mat mx,my;
            FileStorage MX(format("mx%d.xml",0).c_str(), 0);
            MX[format("mx%d",k)] >> mx;
            FileStorage MY(format("my%d.xml",0).c_str(), 0);
            MY[format("my%d",k)] >> my;
            std::vector<Mat> mv {mx,my};
            merge(mv, dst);
        };
        read_map(1, state.mapxL);
        read_map(2, state.mapxR);
    } else {
        state.mapxL = ident(240,376);
        state.mapxR = ident(240,376);
    }
    FileStorage q("Q.xml",0);
    q["Q"] >> state.Q;

    while(1) {

        // capture images from the cameras
        Mat matL, matR;
        matL = imread("ambush_5_left.jpg", 0);
        matR = imread("ambush_5_right.jpg", 0);
        resize(matL,matL,state.mapxL.size());
        resize(matR,matR,state.mapxR.size());

        std::vector<Point3f> pointVector3d;
        std::vector<uchar> pointColors;
        std::vector<Point3i> pointVector2d; // for display

        int64 t0 = getTickCount();
        //pushbroom_stereo.ProcessImages(matL, matR, &pointVector3d, &pointColors, &pointVector2d, state);
        pushbroom::Stereo(matL, matR, pointVector3d, pointColors, pointVector2d, state);
        int64 t1 = getTickCount();

        Mat remapL;
        remap(matL, remapL, state.mapxL, Mat(), INTER_NEAREST);
        imshow("L",remapL);

        Mat remapR;
        remap(matR, remapR, state.mapxR, Mat(), INTER_NEAREST);
        imshow("R",remapR);

        //https://github.com/andybarry/flight/blob/master/sensors/bm-stereo/bm-stereo.cpp#L112-L123
        Ptr<StereoBM> bm = StereoBM::create(3*16, 5);
        bm->setMinDisparity(0);
        bm->setSpeckleWindowSize(59);
        bm->setSpeckleRange(30);

        Mat disparity_bm, disp8;
        int64 t2 = getTickCount();
        bm->compute(remapL, remapR, disparity_bm);
        int64 t3 = getTickCount();
        double T1 = (t1-t0) / getTickFrequency();
        double T2 = (t3-t2) / getTickFrequency();
        cout << "T " << T1 << " / " << T2 << endl;
        disparity_bm.convertTo(disp8, CV_8U);
        cvtColor(disp8,disp8,COLOR_GRAY2BGR);
        for (int i=0; i<(int)pointVector2d.size(); i++) {
            int x = pointVector2d.at(i).x;
            int y = pointVector2d.at(i).y;
            circle(disp8, Point(x, y), state.blockSize*2, Scalar(pointColors[i],10,pointColors[i]), cv::FILLED);
            circle(disp8, Point(x, y), state.blockSize, Scalar(20,20,5*pointColors[i]), cv::FILLED);
        }
        imshow("BM", disp8);

        Mat image_3d;
        reprojectImageTo3D(disparity_bm, image_3d, state.Q, true);
        //cout << "Q " << image_3d.size() << " " << image_3d.type() << endl;
        imshow("Q", image_3d);

        int i=0, off=state.blockSize/2;
        for (auto p:pointVector2d) {
            Point3f a = image_3d.at<Point3f>(p.y+off,p.x+off);
            Point3f b = pointVector3d[i];
            Point3i c = pointVector2d[i++];
            cout << a << " " << b << " " << norm(a-b) << "\t" << c << endl;
        }
        int k = waitKey();
        switch(k) {
            case 27: return 0;
            case 'D': state.disparity ++; cout << "disp " << state.disparity << endl; break;
            case 'd': state.disparity --; cout << "disp " << state.disparity << endl; break;
            case 'Z': state.zero_dist_disparity ++; cout << "zero " << state.zero_dist_disparity << endl; break;
            case 'z': state.zero_dist_disparity --; cout << "zero " << state.zero_dist_disparity << endl; break;
            case 'S': state.sadThreshold ++; cout << "sad  " << state.sadThreshold << endl; break;
            case 's': state.sadThreshold --; cout << "sad  " << state.sadThreshold << endl; break;
            case 'B': state.blockSize ++; cout << "block " << state.blockSize << endl; break;
            case 'b': state.blockSize --; cout << "block " << state.blockSize << endl; break;
        }
    }
    return 0;
}
