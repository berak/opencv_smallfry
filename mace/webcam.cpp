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
        "press 'r' to record images. once N trainimages were recorded, train the mace filter"
        "press 'p' to predict"
        "press 'esc' to return";

int main(int argc, char **argv) {
    CommandLineParser parser(argc, argv,
        "{ help h usage ? ||     show this message }"
        "{ pre p          ||     pretrained mace filter file  (e.g. my.xml) }"
        "{ multi m        ||     use multiple mace filters }"
        "{ num n          |50|   num train images }"
        "{ siz z          |64|   image size }"
        "{ salt s         |0|    if != 0, random convolute with this seed }"
    );
    if (parser.has("help")) {
        parser.printMessage();
        return 1;
    }
    String pre = parser.get<String>("pre");
    int N = parser.get<int>("num");
    int Z = parser.get<int>("siz");
    int S = parser.get<int>("salt");
    int state = NEUTRAL;

    Ptr<MACE> mace;
    if (parser.has("multi"))
        mace = MACE::createSampler(Z,vector<Rect2d>{
            Rect2d(0,0,1,1),
            Rect2d(0.25,0.5,0.5,0.5),
            Rect2d(0,0,0.5,0.5),
            Rect2d(0.5,0,0.5,0.5)
        });
    else
        mace = MACE::create(Z);

    if (! pre.empty()) { // load pretrained model, if available
        FileStorage fs(pre, 0);
        mace->read(fs.root());
        state = PREDICT;
    }

    CascadeClassifier head("haarcascade_frontalface_alt2.xml");
    if (head.empty()) return -1;

    VideoCapture cap(0);
    if (! cap.isOpened()) return -2;

    vector<Mat> tr_img;
    while(1) {
        Mat frame;
        cap >> frame;

        vector<Rect> rects;
        head.detectMultiScale(frame,rects);
        if (rects.size()>0) {
            Rect r = rects[0] & Rect(0,0,frame.cols,frame.rows);
            if (! r.area()) {
                cerr << "inv " << rects[0] << endl;
                continue;
            }
            Scalar col = Scalar(0,120,0);

            if (state == RECORD) {
                if (tr_img.size() >= N) {
                    mace->salt(S);
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
                else      col = Scalar(60,60,60);
            }

            rectangle(frame, rects[0], col, 2);
        }

        imshow("MACE",frame);
        int k = waitKey(10);
        switch (k) {
            case -1 : break;
            case 27 : return 0;
            default : state = NEUTRAL; break;
            case 'r': state = RECORD;  break;
            case 'p': state = PREDICT; break;
            case 's': { FileStorage fs("my.xml",1); mace->write(fs); break; }
        }
    }

    return 1;
}
