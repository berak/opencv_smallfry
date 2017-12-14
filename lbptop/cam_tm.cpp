#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>
#include <deque>

using namespace cv;
using namespace std;


struct TrackerTM {
    Mat templ;
    Rect box,bounds;

    bool init(const Mat &img, const Rect &bbox) {
        if (bbox.empty()) return false;
        templ = img(bbox);
        bounds = Rect(0, 0, img.cols, img.rows);
        box = bbox & bounds; // clip
        return true;
    }

    bool update(const Mat &img, Rect &bbox) {
        if (templ.empty() || box.empty()) return false;
        Rect search(box.x-box.width/2, box.y-box.height/2, box.width*2, box.height*2);
        search &= bounds;
        Mat result;
        matchTemplate(img(search), templ, result, CV_TM_SQDIFF_NORMED);
        Point p; double v;
        minMaxLoc(result, &v, 0, &p, 0);
        imshow("!", result);
        box.x = search.x + p.x; // from search space to img space
        box.y = search.y + p.y;
        box &= bounds; // clip again
        bbox = box;
        return v < 0.03; // experimental threshold
    }
};

// -------8<------------------------------------------------------------------

int main(int argc, const char* argv[]) {
    //VideoCapture cap("C:\\p\\opencv_contrib\\modules\\tracking\\samples\\david.webm");
    VideoCapture cap(0);
    CascadeClassifier cad("haarcascade_frontalface_alt.xml");
    TrackerTM tm;

    while(cap.isOpened()) {
        Mat frame, gray;
        cap >> frame;
        cv::resize(frame,frame,cv::Size(frame.cols/2, frame.rows/2));

        Rect box;
        bool ok = tm.update(frame, box);

        rectangle(frame, box, (ok ? Scalar(0,200,0) : Scalar(0,0,200)), 1);
        imshow("tracker",frame);
        int k = waitKey(10);

        if (k == 27) return 0; // esc. pressed
        if (!ok || k == ' ') { // space pressed, or obj lost
            cv::cvtColor(frame, gray, CV_BGR2GRAY);
            vector<Rect> faces;
            cad.detectMultiScale(gray,faces,1.1,4,CV_HAAR_FIND_BIGGEST_OBJECT,cv::Size(30,30));
            if (faces.size()) {
                box = faces[0];
                box.height *= 1.05;
                tm.init(frame, box);
            }
        }
    }
    return 0;
}

