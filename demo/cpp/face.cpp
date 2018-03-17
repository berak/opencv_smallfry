#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

int main(int argc, const char *argv[]) {
    string fn_haar = "haarcascade_frontalface_alt.xml";
    string fn_vid = "c:/data/video/Identification.avi";


    Ptr<FaceRecognizer> model = LBPHFaceRecognizer::create();
    model->setThreshold(130);

    CascadeClassifier haar_cascade(fn_haar);

    VideoCapture cap(fn_vid);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture " << fn_vid << "cannot be opened." << endl;
        return -1;
    }

    //
    // pass 1 : label images & update model:
    //
    Mat frame;
    int nf=0;
    for(;;) {
        int p = cap.get(CAP_PROP_POS_FRAMES);
        cap.set(CAP_PROP_POS_FRAMES, p+3); // speed it up a bit while training
        nf += 3;
        cap >> frame;
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces, 1.1, 4);

        for(size_t i = 0; i < faces.size(); i++) {
            Mat original = frame.clone();
            Rect face_i = faces[i];
            Mat face = gray(face_i);
            rectangle(original, face_i, Scalar(255,0,0), 1);
            imshow("face_recognizer", original);
            int k = waitKey();
            int id = -1;
            switch(k) {
                case '0': id=0; break;
                case '1': id=1; break;
                case '2': id=2; break;
                case '3': id=3; break;
                case '4': id=4; break;
                case '5': id=5; break;
                case '6': id=6; break;
                case '7': id=7; break;
                case '8': id=8; break;
                case '9': id=9; break;
                case 'q': id=10; break;
                case 'w': id=11; break;  // german keyboard here ;()
                case 'e': id=12; break;
                case 'r': id=13; break;
                case 't': id=14; break;
                case 'z': id=15; break;
                case 'u': id=16; break;
                case 'i': id=17; break;
                case 'o': id=18; break;
                case 'p': id=19; break;
                default: { // cue
                    int p = cap.get(CAP_PROP_POS_FRAMES);
                    cap.set(CAP_PROP_POS_FRAMES, p+50);
                    nf += 50;
                    break;
                }
                case 8: { // backspace
                    int p = cap.get(CAP_PROP_POS_FRAMES);
                    cap.set(CAP_PROP_POS_FRAMES, p-50);
                    nf -= 50;
                    break;
                }
                case ' ': goto test;
            }
            if (id != -1) {
                vector<Mat> images{face};
                vector<int> labels{id};
                model->update(images, labels);
                cerr << nf << " added: " << id << endl;
            }
        }
        // And display it:
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }

    //
    //
    // pass 2: test trained model
    //
    //
test:
    cerr << "saving model." << endl;
    model->save("lbpfaces.xml.gz");
    cap.set(CAP_PROP_POS_FRAMES, 0);
    for(;;) {
        cap >> frame;
        // Clone the current frame:
        Mat original = frame.clone();
        // Convert the current frame to grayscale:
        Mat gray;
        cvtColor(original, gray, COLOR_BGR2GRAY);

        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);

        for(size_t i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];
            Mat face = gray(face_i);
            int prediction = model->predict(face);
            string box_text = format("Prediction = %d", prediction);
            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
            rectangle(original, face_i, Scalar(255,0,0), 1);
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, Scalar(255,0,0), 2);
        }
        // Show the result:
        imshow("face_recognizer", original);
        // And display it:
        char key = (char) waitKey(10);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }
    return 0;
}
