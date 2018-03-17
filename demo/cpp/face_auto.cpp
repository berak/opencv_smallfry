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
    model->setThreshold(60);

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
    vector<int> counter(200,0);
    int nf=0;
    int id=0;
    for(;;) {
        int p = cap.get(CAP_PROP_POS_FRAMES);
        cap.set(CAP_PROP_POS_FRAMES, p+2); // speed it up a bit while training
        nf += 2;
        cap >> frame;
        if (frame.empty()) break;
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces, 1.1, 4);

        Mat original = frame.clone(); // copy for drawing
        for(size_t i = 0; i < faces.size(); i++) {
            Rect face_i = faces[i];
            Mat face = gray(face_i);
            int prediction = id;
            if (model->empty()) { // special case, untrained model
                vector<Mat> images{face};
                vector<int> labels{id};
                model->update(images, labels);
                counter[prediction] ++;
                cerr << nf << " " << prediction << " " << counter[prediction] << endl;
            } else {
                double conf=-1;
                model->predict(face, prediction, conf);
                if (prediction == -1 ) {
                    prediction = id;
                    id++;
                }
                if (counter[prediction] < 20) {
                    vector<Mat> images{face};
                    vector<int> labels{prediction};
                    model->update(images, labels);
                    counter[prediction] ++;
                }
                cerr << nf << " " << prediction << " " << counter[prediction] << " " << conf << endl;
            }
            string box_text = format("%d", prediction);
            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
            rectangle(original, face_i, Scalar(255,0,0), 1);
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.5, Scalar(255,0,0), 2);
        }
        // And display it:
        imshow("face_recognizer", original);
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }

    return 0;
}
