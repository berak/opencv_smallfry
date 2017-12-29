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
        "press 'r' to record images. once N trainimages were recorded, train the mace filter\n"
        "press 'p' to predict (twofactor mode will switch back to neutral after each prediction attempt)\n"
        "press 's' to save a trained model\n"
        "press 'esc' to return\n"
        "any other key will reset to neutral state\n";

int main(int argc, char **argv) {
    CommandLineParser parser(argc, argv,
        "{ help h usage ? ||     show this message }"
        "{ pre p          ||     pretrained mace filter file  (e.g. my.xml) }"
        "{ num n          |50|   num train images }"
        "{ size s         |64|   image size }"
        "{ twofactor t    ||     pass phrase(text) for 2 factor authentification.\n"
        "                     (random convolute images seeded with the crc of this)\n"
        "                     users will get prompted to guess the secrect, additional to the image. }"
    );
    if (parser.has("help")) {
        parser.printMessage();
        return 1;
    } else {
        cout << help << endl;
    }
    String defname = "my.xml.gz";
    String pre = parser.get<String>("pre");
    String two = parser.get<String>("twofactor");
    int N = parser.get<int>("num");
    int Z = parser.get<int>("size");
    int state = NEUTRAL;

    Ptr<MACE> mace = MACE::create(Z);

    if (! two.empty()) {
        cout << "'" << two << "' initial passphrase" << endl;
        mace->salt(two);
    }

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
            Scalar col = Scalar(0,120,0);

            if (state == RECORD) {
                if (tr_img.size() >= N) {
                    mace->train(tr_img);
                    tr_img.clear();
                    state = PREDICT;
                } else {
                    tr_img.push_back(frame(rects[0]).clone());
                }
                col = Scalar(200,0,0);
            }

            if (state == PREDICT) {
                if (! two.empty()) { // prompt for secret on console
                    cout << "enter passphrase: ";
                    string pass;
                    getline(cin, pass);
                    mace->salt(pass);
                    state = NEUTRAL;
                    cout << "'" << pass << "' : ";
                }
                bool same = mace->same(frame(rects[0]));
                if (same) col = Scalar(0,220,220);
                else      col = Scalar(60,60,60);
                if (! two.empty()) {
                    cout << (same ? "accepted." : "denied.") << endl;
                }
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
            case 's': { FileStorage fs(defname, 1); mace->write(fs); break; }
        }
    }

    return 0;
}
