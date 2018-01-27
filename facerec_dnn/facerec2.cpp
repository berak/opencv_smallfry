//
// realtime face recognition/transfer learning demo (using facenet features).
// you'll need the pretrained facenet dnn model from here (30.1mb):
//    https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7
//

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;


#include "profile.h"
// default face size, 96 - as required by the pretrained facenet
const int  FIXED_FACE = 96;
// restrict feature count per person
const int MAX_IMAGES_PP = 100;

// program states
enum
{
    NEUTRAL = 0,
    RECORD  = 1,
    PREDICT = 2
};

// our database, each row in the Mat is a facenet feature
typedef map<String,Mat> Person;

class FaceNet
{
    dnn::Net net;
    Person persons;

public:

    FaceNet(const String &pathToTorchnet)
    {
        try {
            net = dnn::readNetFromTorch(pathToTorchnet);
        } catch(Exception &e) {
            cerr << "Download it from:  ";
            cerr << "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7" << endl;
        }
    }

    String info(bool detailled=false) {
        String inf = "";
        int n=0;
        for (Person::iterator it=persons.begin(); it != persons.end(); ++it) {
            if (detailled) inf += format("%16s %d\n", it->first.c_str(), it->second.rows);
            n += it->second.rows;
        }
        inf += format("%d persons, %d features, %dk floats\n", persons.size(), n, (n*128)/1024);
        return inf;
    }

    Mat process(const Mat &image)
    {
        PROFILE
        Mat inputBlob = dnn::blobFromImage(image, 1./255, Size(FIXED_FACE, FIXED_FACE), Scalar(), true, false);
        net.setInput(inputBlob);
        return net.forward();
    }

    bool addPerson(const String &p, const vector<Mat> &images)
    {
        PROFILE
        Mat &pers = persons[p];
        for (const Mat &m : images) {
            Mat f = process(m);
            if (pers.rows < MAX_IMAGES_PP) {
                pers.push_back(f.reshape(1,1));
            } else {
                size_t i=theRNG().uniform(0,MAX_IMAGES_PP);
                f.copyTo(pers.row(i));
            }
        }
        return true;
    }

    bool delPerson(const String &p)
    {
        Person::iterator pers = persons.find(p);
        if (pers==persons.end()) return false;
        persons.erase(pers);
        return true;
    }

    String predict(const Mat & img)
    {
        PROFILEX("predict")
        Mat feature = process(img);
        String best = "";
        double mind = 1e6, maxd = 0;
        for (Person::iterator it=persons.begin(); it != persons.end(); ++it) {
            Mat &f = it->second;
            for (int r=0; r<f.rows; r++) {
                double d = norm(feature, f.row(r));
                if (d < mind) {
                    mind = d;
                    best = it->first;
                }
                maxd = max(d, maxd);
            }
        }
        double score = 1.0 - mind / maxd;
        return best + format("  %3.2f", score);
    }

    void write(FileStorage& sav) const
    {
        PROFILE
        sav << "size" << int(persons.size());
        sav << "persons" << "[";
        for (map<String,Mat>::const_iterator it = persons.begin(); it!=persons.end(); ++it) {
            sav << "{" << "k" << it->first << "v" << it->second << "}";
        }
        sav << "]";
    }
    void read(const FileNode& fs)
    {
        PROFILE
        FileNode n = fs["persons"];
        FileNodeIterator it = n.begin(), it_end = n.end();
        for (; it != it_end; ++it) {
            String k; (*it)["k"] >> k;
            Mat v; (*it)["v"] >> v;
            persons[k]=v;
        }
    }
    void save(const String &s) {
        PROFILE
        FileStorage fs(s, FileStorage::WRITE | FileStorage::BASE64);
        write(fs);
        fs.release();
    }
    void load(const String &s) {
        PROFILE
        FileStorage fs(s, 0);
        if (!fs.isOpened()) return;
        read(fs.root());
        fs.release();
    }
};

int main(int argc, const char *argv[])
{
    CommandLineParser parser(argc, argv,
        "{h help    || show this help screen}"
        "{v video   |0| video path (a filename, 0 or 1 to use webcam)}"
        "{m model   |C:/data/mdl/openface.nn4.small2.v1.t7| path to the trained network model}"
        "{c cascade |C:/data/mdl/haarcascade_frontalface_alt.xml| path to the haarcascade for face detection}"
    );
    String cp = parser.get<String>("video");
    String torchModel = parser.get<String>("model");
    String cascade_path = parser.get<String>("cascade");
    if (parser.has("help") || torchModel.empty() || cascade_path.empty()) {
        parser.printMessage();
        return 0;
    }

    cv::CascadeClassifier cascade(cascade_path);
    if (cascade.empty()) {
        cerr << "Could not load cascade: " << cascade_path  << endl;
        return -1;
    }

    VideoCapture cap;
    if (cp == "0") cap.open(0);
    else if (cp == "1") cap.open(1);
    else cap.open(cp);
    if (! cap.isOpened()) {
        cerr << "Could not open capture: " << cp << endl;
        return -2;
    }

    FaceNet reco(torchModel);
    reco.load("facedb.xml");

    cout << "press 'r'      to record new persons" << endl;
    cout << "      'space'  to stop recording (then input a name on the console)" << endl;
    cout << "      'p'      to predict" << endl;
    cout << "      'esc'    to quit" << endl;

    vector<Mat> images;
    String caption = "";
    int frameNo = 0;
    int state = NEUTRAL;
    Scalar color[3] = {
        Scalar(30,100,30),
        Scalar(10,10,160),
        Scalar(200,130,30),
    };

    while(true) {
        PROFILEX("frame")
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        Mat gray;
        cvtColor(frame, gray, COLOR_RGB2GRAY);

        std::vector<cv::Rect> faces;
        { PROFILEX("detect")
        cascade.detectMultiScale(gray, faces, 1.2, 3,
            CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH  ,
            Size(40, 40), Size(300,300));
        }
        if (faces.size() > 0) {
            Rect roi = faces[0];
            if ((state == RECORD ) && (frameNo % 3 == 0)) {
                images.push_back(frame(roi).clone());
                cout << ".";
            }
            if (state == PREDICT) {
                caption = reco.predict(frame(roi));
                if (caption != "") {
                    putText(frame, caption, Point(roi.x, roi.y+roi.width+13),
                        FONT_HERSHEY_PLAIN, 1.3, color[state], 1, LINE_AA);
                }
            }
            rectangle(frame, roi, color[state]);
        }
        for(int i=6,sc=6; i>1; i--,sc+=2) // status led
            circle(frame, Point(10,10), i, color[state]*(float(sc)/10), -1, LINE_AA);

        imshow("Face Recognition", frame);
        int k = waitKey(30);
        if (k == 27 ) break;
        if (k == 'i') cout << reco.info(true) << endl;
        if (k == 'p') state = PREDICT;
        if (k == 'r' && state != RECORD ) {
            images.clear();
            state = RECORD ;
        }
        if (k==' ') {
            if ((state == RECORD ) && (!images.empty())) {
                // ask for a name, and write the images to that folder:
                cout << endl << "please enter a name (leave empty to ignore) :" << endl;
                string n; cin >> n;
                if ((!n.empty()) && (images.size() > 0)) {
                    reco.addPerson(n, images);
                    reco.save("facedb.xml");
                }
            }
            state = NEUTRAL;
        }
        frameNo++;
    }
    return 0;
}
