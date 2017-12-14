#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>
#include <deque>
#include "profile.h"

using namespace cv;
using namespace std;


struct TrackerCamShift {
    int smin, smax, vmin, vmax;
    Mat hue, mask, hist;
    Rect box;

    TrackerCamShift(int smin=30, int smax=256, int vmin=10, int vmax=256)
        : smin(smin), smax(smax), vmin(vmin), vmax(vmax)
    {}

    void getHue(const Mat &img) {
        Mat hsv;
        cvtColor(img, hsv, COLOR_BGR2HSV);
        inRange(hsv, Scalar(0, smin, vmin), Scalar(180, smin, vmax), mask);
        extractChannel(hsv, hue, 0);
    }

    bool init(const Mat &img, const Rect &bbox) {
        getHue(img);
        Mat roi(hue, bbox), maskroi(mask, bbox);
        int hsize = 16;
        float hranges[] = {0,180};
        const float* phranges = hranges;
        calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
        normalize(hist, hist, 0, 255, NORM_MINMAX);
        box = bbox;
    }

    bool update(const Mat &img, Rect &bbox) {
        getHue(img);
        Mat backproj;
        float hranges[] = {0,180};
        const float* phranges = hranges;
        calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
        backproj &= mask;
        RotatedRect trackBox = CamShift(backproj, box, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
        bbox = box = trackBox.boundingRect();
    }
};
struct TrackerTM {
    Mat templ;
    Rect box,bounds;

    bool init(const Mat &img, const Rect &bbox) {
        templ = img(bbox);
        box = bbox;
        bounds = Rect(0, 0, img.rows, img.cols);
    }

    bool update(const Mat &img, Rect &bbox) {
        PROFILE;
        Rect search(box.x-box.width/2, box.y-box.height/2, box.width*2, box.height*2);
        search &= bounds;
        Mat result;
        matchTemplate(img, templ, result, CV_TM_SQDIFF);
        Point p;
        minMaxLoc(result, 0, 0, &p, 0);
        box.x = p.x;
        box.y = p.y;
        bbox = box & bounds;
    }
};

// -------8<------------------------------------------------------------------

int main(int argc, const char* argv[]) {
    VideoCapture cap(0);
    CascadeClassifier cad("haarcascade_frontalface_alt.xml");

        // >>>> Kalman Filter
    int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;

    unsigned int type = CV_32F;
    cv::KalmanFilter KF(stateSize, measSize, contrSize, type);

    cv::Mat state(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
    cv::Mat measurement(measSize, 1, type);    // [z_x,z_y,z_w,z_h]
    //cv::Mat procNoise(stateSize, 1, type)
    // [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

    // Transition State Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
    cv::setIdentity(KF.transitionMatrix);

    // Measure Matrix H
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]
    KF.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    KF.measurementMatrix.at<float>(0) = 1.0f;
    KF.measurementMatrix.at<float>(7) = 1.0f;
    KF.measurementMatrix.at<float>(16) = 1.0f;
    KF.measurementMatrix.at<float>(23) = 1.0f;

    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]
    //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
    KF.processNoiseCov.at<float>(0) = 1e-2;
    KF.processNoiseCov.at<float>(7) = 1e-2;
    KF.processNoiseCov.at<float>(14) = 5.0f;
    KF.processNoiseCov.at<float>(21) = 5.0f;
    KF.processNoiseCov.at<float>(28) = 1e-2;
    KF.processNoiseCov.at<float>(35) = 1e-2;

    // Measures Noise Covariance Matrix R
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar(1e-1));

    TrackerCamShift cs;
    TrackerTM tm;
    bool doKalman = false;
    bool doCamShift = false;
    bool doTM = false;

    while(cap.isOpened()) {
        Mat frame, gray;
        cap >> frame;
        cv::resize(frame,frame,cv::Size(frame.cols/2, frame.rows/2));
        cv::cvtColor(frame,gray,CV_BGR2GRAY);
        cv::equalizeHist(gray,gray);

        Rect box, screen(0, 0, frame.cols, frame.rows);
        vector<Rect> faces;
        cad.detectMultiScale(gray,faces,1.1,4,CV_HAAR_FIND_BIGGEST_OBJECT,cv::Size(30,30));
        if (faces.size()) {
            box = faces[0];
            box.height *= 1.05;
            Mat prediction = KF.predict();
            float px = prediction.at<float>(0);
            float py = prediction.at<float>(1);
            float pw = prediction.at<float>(4);
            float ph = prediction.at<float>(5);
            measurement.at<float>(0) = float(box.x);
            measurement.at<float>(1) = float(box.y);
            measurement.at<float>(2) = float(box.width);
            measurement.at<float>(3) = float(box.height);
            KF.correct(measurement);
            if (doKalman) {
                cout << box << " " << px << " " << py << " " << pw << " " << ph << endl;
                box.x = px;
                box.y = py;
                box.width = pw;
                box.height = ph;
            }
            if (doCamShift) {
                cs.update(frame, box);
               // RotatedRect r = CamShift(img,)
            }
            if (doTM) {
                tm.update(frame, box);
            }
            box &= screen;
            rectangle(frame, box, Scalar(0,200,0), 1);
        }
        imshow("TOP",frame);
        int k = waitKey(10);
        switch(k) {
            case 27: return 0;
            case ' ': doKalman = ! doKalman; cout << "kalman: " << doKalman << endl; break;
            case 'c': {
                doCamShift = ! doCamShift;
                if (doCamShift)
                    cs.init(frame, box);
                cout << "CamShift: " << doCamShift << endl;
                break;
            }
            case 't': {
                doTM = ! doTM;
                if (doTM)
                    tm.init(frame, box);
                cout << "TMatch: " << doTM << endl;
                break;
            }
        }
    }
    return 0;
}

