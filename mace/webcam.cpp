#include "opencv2/opencv.hpp"
#include "mace.h"

using namespace cv;
using namespace std;

enum STATE {
    NEUTRAL,
    RECORD,
    PREDICT
};

const char *help =
        "press 'r' to record images. once num trainimages were recorded, train the mace filter"
        "press 'p' to predict"
        "press 'esc' to return";

int main(int argc, char **argv) {
    CommandLineParser parser(argc, argv,
        "{ help h usage ? ||     show this message }"
        "{ pre p          ||     pretrained mace filter file  (e.g. my.xml) }"
        "{ num n          |50|   num train images }"
        "{ siz z          |64|   image size }"
        "{ salt s         |0|    if != 0, random convolute with this seed }"
    );
    if (parser.has("help")) {
        parser.printMessage();
        return 1;
    }
    String pre = parser.get<String>("pre");
    int Z = parser.get<int>("siz");
    int S = parser.get<int>("salt");
    int state = NEUTRAL;

    Ptr<MACE> mace = MACE::create(Z, S);
    if (!pre.empty()) {
        FileStorage fs(pre, 0);
        mace->read(fs.root());
        state = PREDICT;
    }

    CascadeClassifier head("haarcascade_frontalface_alt2.xml");
    if (head.empty()) return -1;

    VideoCapture cap(0);
    if (!cap.isOpened()) return -2;

    vector<Mat> tr_img;
    while(1) {
        Mat frame;
        cap >> frame;

        vector<Rect> rects;
        head.detectMultiScale(frame,rects);
        if (rects.size()>0) {
            Scalar col = Scalar(0,120,0);

            if (state == RECORD) {
                if (tr_img.size() > 50) {
                    mace->train(tr_img);
                    tr_img.clear();
                    state = PREDICT;
                } else {
                    tr_img.push_back(frame(rects[0]).clone());
                }
                col = Scalar(200,0,0);
            }

            if (state == PREDICT) {
                bool same = mace->same(frame(rects[0]));
                if (same) col = Scalar(0,220,220);
            }

            rectangle(frame, rects[0], col, 2);
        }

        imshow("M",frame);
        int k = waitKey(10);
        if (k >  1) state = NEUTRAL;
        if (k== 27) break;
        if (k=='r') state = RECORD;
        if (k=='p') state = PREDICT;
        if (k=='s') { FileStorage fs("my.xml",1); mace->write(fs); }
    }

    return 1;
}
