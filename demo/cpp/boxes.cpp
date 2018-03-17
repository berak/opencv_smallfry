#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/line_descriptor.hpp"

using namespace cv;
using namespace std;

// intersection over union ratio
bool match(const Rect &a, const Rect &b) {
    float iou = float((a & b).area()) / (a | b).area();
    return iou > 0.8f;
}

struct box {
    Rect r;
    int seen;
};


int main() {
    String face_cascade_name =  "c:/p/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
    CascadeClassifier face_cascade(face_cascade_name);
    VideoCapture cap(0);
    std::vector<box> boxes;
    while(1) {
        Mat frame;
        if (! cap.read(frame) ) break;

        std::vector<Rect> faces;
        face_cascade.detectMultiScale(frame, faces, 1.1, 0, 0);
        cerr << faces.size() << " faces detected, " << boxes.size() << " boxes ." << endl;
        for ( size_t i=0; i<faces.size(); i++) {
            rectangle( frame, faces[i], Scalar(0,255,0), 1);
            if (boxes.size()) {
                for (size_t j=0; j<boxes.size(); j++) {
                    if (match(faces[i], boxes[j].r)) {
                        boxes[j].r = faces[i]; // adjust position
                        boxes[j].seen += 2;
                        break;
                    }
                }
            } else {
                box b;
                b.r = faces[i];
                b.seen = 2;
                boxes.push_back(b);
            }
        }
        for (std::vector<box>::iterator it = boxes.begin(); it != boxes.end(); ) {
            it->seen --;
            if (it->seen <= 0) {
                it = boxes.erase(it);
            } else {
                rectangle( frame, it->r, Scalar( 255, 0, 255 ), 4, 8, 0 );
                cerr << it->r << " " << it->seen << endl;
                it++;
            }
        }
        cv::imshow("src", frame);
        if (cv::waitKey(10) == 27) break;
    }
    return 0;
}
